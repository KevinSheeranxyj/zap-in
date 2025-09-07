use anchor_lang::prelude::*;
use anchor_lang::solana_program::pubkey;
use anchor_spl::token::{self, Mint, Token, TokenAccount, Transfer};
use anchor_spl::token_interface::{Token2022, Mint as InterfaceMint, TokenAccount as InterfaceTokenAccount};
use anchor_spl::metadata::Metadata;
use anchor_lang::prelude::Rent;
use anchor_spl::memo::spl_memo;
use anchor_lang::system_program;
use anchor_lang::prelude::Sysvar;
use anchor_lang::error::Error;
use raydium_amm_v3::libraries::{big_num::*, full_math::MulDiv, tick_math};
use anchor_spl::associated_token::AssociatedToken;
use std::str::FromStr;
use anchor_lang::solana_program::sysvar;
use raydium_amm_v3::{
    cpi,
    program::AmmV3,
    states::{PoolState, AmmConfig, POSITION_SEED, TICK_ARRAY_SEED, ObservationState, TickArrayState, ProtocolPositionState, PersonalPositionState},
};
use anchor_spl::associated_token::get_associated_token_address_with_program_id;
use anchor_lang::solana_program::hash::hash as solana_hash;
use anchor_lang::solana_program::{
    program::invoke_signed,
    program_pack::Pack,
    system_instruction,
};
use anchor_spl::token::spl_token;

declare_id!("9T7YMp5SXZvP3nqUj9B7rQGFErfMmh8t59jvxtV3CnjB");

/// NOTE: For ZapIn & ZapOut, we're leveraging the Raydium-Amm-v3 Protocol SDK to robost our requirement
#[program]
pub mod dg_solana_zapin {
    use super::*;

    pub const RAYDIUM_CLMM_PROGRAM_ID: Pubkey =
        pubkey!("CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"); // mainnet program ID

    // Initialize the PDA and set the authority
    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        let operation_data = &mut ctx.accounts.operation_data;
        operation_data.authority = ctx.accounts.authority.key();
        operation_data.initialized = true;
        msg!("Initialized PDA with authority: {}", operation_data.authority);
        Ok(())
    }

    #[event]
    pub struct DepositEvent {
        pub transfer_id: String,
        pub amount: u64,
        pub recipient: Pubkey,
    }

    // Deposit transfer details into PDA
    pub fn deposit(
        ctx: Context<Deposit>,
        transfer_id: String,
        operation_type: OperationType,
        action: Vec<u8>,
        amount: u64,
        ca: Pubkey,
    ) -> Result<()> {
        let od = &mut ctx.accounts.operation_data;

        // 初始化（首次该 transfer_id）
        if !od.initialized {
            od.authority = ctx.accounts.authority.key();
            od.initialized = true;
            msg!("Initialized operation_data for transfer_id {} with authority {}", transfer_id, od.authority);
        }

        require!(amount > 0, OperationError::InvalidAmount);
        require!(!transfer_id.is_empty(), OperationError::InvalidTransferId);

        // 资金转入（保持原逻辑）
        let cpi_accounts = Transfer {
            from: ctx.accounts.authority_ata.to_account_info(),
            to: ctx.accounts.program_token_account.to_account_info(),
            authority: ctx.accounts.authority.to_account_info(),
        };
        let cpi_program = ctx.accounts.token_program.to_account_info();
        let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
        token::transfer(cpi_ctx, amount)?;

        // 存基础参数
        od.transfer_id = transfer_id.clone();
        od.amount = amount;
        od.executed = false;
        od.ca = ca;
        od.operation_type = operation_type.clone();
        od.action = action.clone(); // 保留原始参数

        // ====== 存 Raydium 固定账户（直接从 ctx 读 pubkey）======
        od.clmm_program_id   = ctx.accounts.clmm_program.key();
        od.pool_state        = ctx.accounts.pool_state.key();
        od.amm_config        = ctx.accounts.amm_config.key();
        od.observation_state = ctx.accounts.observation_state.key();
        od.token_vault_0     = ctx.accounts.token_vault_0.key();
        od.token_vault_1     = ctx.accounts.token_vault_1.key();
        od.token_mint_0      = ctx.accounts.token_mint_0.key();
        od.token_mint_1      = ctx.accounts.token_mint_1.key();

        // 如果是 ZapIn，解析参数并派生 tick array / protocol_position 等，存起来
        if let OperationType::ZapIn = operation_type {
            let p: ZapInParams = deserialize_params(&od.action)?;
            od.tick_lower = p.tick_lower;
            od.tick_upper = p.tick_upper;

            // 根据 pool 的 tick_spacing 计算 tick array 起始
            let pool = ctx.accounts.pool_state.load()?;
            let tick_spacing: i32 = pool.tick_spacing.into();
            let lower_start = tick_array_start_index(p.tick_lower, tick_spacing);
            let upper_start = tick_array_start_index(p.tick_upper, tick_spacing);

            // Raydium tick array PDA（由外部提供，但我们把“应有地址”存起来用作后续校验）
            let (ta_lower, _) = Pubkey::find_program_address(
                &[
                    TICK_ARRAY_SEED.as_bytes(),
                    ctx.accounts.pool_state.key().as_ref(),
                    &lower_start.to_be_bytes(),
                ],
                &ctx.accounts.clmm_program.key(),
            );
            let (ta_upper, _) = Pubkey::find_program_address(
                &[
                    TICK_ARRAY_SEED.as_bytes(),
                    ctx.accounts.pool_state.key().as_ref(),
                    &upper_start.to_be_bytes(),
                ],
                &ctx.accounts.clmm_program.key(),
            );
            od.tick_array_lower = ta_lower;
            od.tick_array_upper = ta_upper;

            // 协议仓位 PDA（Raydium POSITION_SEED, pool, lower_start, upper_start）
            let (pp, _) = Pubkey::find_program_address(
                &[
                    POSITION_SEED.as_bytes(),
                    ctx.accounts.pool_state.key().as_ref(),
                    &lower_start.to_be_bytes(),
                    &upper_start.to_be_bytes(),
                ],
                &ctx.accounts.clmm_program.key(),
            );
            od.protocol_position = pp;

            // Position NFT mint（deposit 阶段未持有 user；先置空，execute 再写）
            od.position_nft_mint = Pubkey::default();
        }

        // 如果是 Transfer，存 recipient
        if let OperationType::Transfer = operation_type {
            let p: TransferParams = deserialize_params(&od.action)?;
            od.recipient = p.recipient;
        }

        emit!(DepositEvent { transfer_id, amount, recipient: od.recipient });
        Ok(())
    }

    // Execute the token transfer (ZapIn only)
    pub fn execute(ctx: Context<Execute>, transfer_id: String) -> Result<()> {
        // 基础校验（只读借用，立刻结束）
        {
            let od_ref = &ctx.accounts.operation_data;
            require!(od_ref.initialized, OperationError::NotInitialized);
            require!(!od_ref.executed, OperationError::AlreadyExecuted);
            require!(od_ref.amount > 0, OperationError::InvalidAmount);
            require!(od_ref.transfer_id == transfer_id, OperationError::InvalidTransferId);
            require!(matches!(od_ref.operation_type, OperationType::ZapIn), OperationError::InvalidParams);
        }

        // 拷出关键字段（结束对 od 的借用）
        let (
            pool_state_key, amm_config_key, observation_key,
            vault0_key, vault1_key, mint0_key, mint1_key,
            tick_array_lower_key, tick_array_upper_key,
            protocol_pos_key, ca_mint,
            amount_total,
            mut pos_mint, personal_pos_stored,
            action_bytes, pool_key_for_pos_mint,
        ) = {
            let od = &ctx.accounts.operation_data;
            (
                od.pool_state, od.amm_config, od.observation_state,
                od.token_vault_0, od.token_vault_1, od.token_mint_0, od.token_mint_1,
                od.tick_array_lower, od.tick_array_upper,
                od.protocol_position, od.ca,
                od.amount,
                od.position_nft_mint, od.personal_position,
                od.action.clone(), od.pool_state,
            )
        };
        let clmm_pid = { let od = &ctx.accounts.operation_data; od.clmm_program_id };
        let user_key = ctx.accounts.user.key();
        let od_key   = ctx.accounts.operation_data.key();

        // signer seeds
        let bump = ctx.bumps.operation_data;
        let h = transfer_id_hash_bytes(&transfer_id);
        let signer_seeds_slice: [&[u8]; 3] = [b"operation_data".as_ref(), h.as_ref(), &[bump]];
        let signer_seeds: &[&[&[u8]]] = &[&signer_seeds_slice];

        // 关键：不要 clone / to_vec；直接用同一 'info 的切片
        {
            let ras = ctx.remaining_accounts;

            let clmm_prog_ai = find_acc(ras, &clmm_pid, "clmm_program")?;      // od.clmm_program_id
            let token_prog_ai = find_acc(ras, &token::ID, "token_program")?;
            let token22_prog_ai = find_acc(ras, &Token2022::id(), "token_program_2022")?;
            let memo_prog_ai = find_acc(ras, &spl_memo::id(), "memo_program")?;
            let system_prog_ai = find_acc(ras, &system_program::ID, "system_program")?;
            let rent_sysvar_ai = find_acc(ras, &sysvar::rent::id(), "rent_sysvar")?;
            let user_ai = find_acc(ras, &user_key, "user")?;
            let operation_ai = find_acc(ras, &od_key, "operation_data_pda")?;


            // pool/config/observation + vaults + mints
            let pool_state = find_acc(ras, &pool_state_key, "pool_state")?;
            let amm_config = find_acc(ras, &amm_config_key, "amm_config")?;
            let observation = find_acc(ras, &observation_key, "observation_state")?;
            let vault0 = find_acc(ras, &vault0_key, "token_vault_0")?;
            let vault1 = find_acc(ras, &vault1_key, "token_vault_1")?;
            let mint0 = find_acc(ras, &mint0_key, "token_mint_0")?;
            let mint1 = find_acc(ras, &mint1_key, "token_mint_1")?;

            // tick arrays & protocol position
            let ta_lower = find_acc(ras, &tick_array_lower_key, "tick_array_lower")?;
            let ta_upper = find_acc(ras, &tick_array_upper_key, "tick_array_upper")?;
            let protocol_pos = find_acc(ras, &protocol_pos_key, "protocol_position")?;

            // PDA-owned input/output token accounts (mint0/mint1)
            let pda_input_ata = find_pda_token_by_mint(ras, &od_key, &mint0_key, "pda_input_token_account")?;
            let pda_output_ata = find_pda_token_by_mint(ras, &od_key, &mint1_key, "pda_output_token_account")?;

            // program_token_account (the deposit was made here; owner = operation_data PDA)
            let program_token_account = {
                let ai = ras.iter().find(|ai| {
                    let Ok(data_ref) = ai.try_borrow_data() else { return false; };
                    if data_ref.len() < spl_token::state::Account::LEN { return false; }
                    if let Ok(acc) = spl_token::state::Account::unpack_from_slice(&data_ref) {
                        acc.owner == od_key && (acc.mint == mint0_key || acc.mint == mint1_key)
                    } else {
                        false
                    }
                }).ok_or_else(|| error!(OperationError::InvalidParams))?;
                ai.clone()
            };

            // refund recipient token account (user ATA of the same mint as program_token_account.mint)
            let program_token_mint = unpack_token_account(&program_token_account)
                .ok_or_else(|| error!(OperationError::InvalidParams))?
                .mint;
            let recipient_refund_ata = find_user_token_by_mint(
                ras,
                &user_key,
                &program_token_mint,
                "recipient_refund_ata",
            )?;

            // ---- position NFT mint (PDA of *this* program) & user NFT ATA ----
            if pos_mint == Pubkey::default() {
                let (m, _) = Pubkey::find_program_address(
                    &[b"pos_nft_mint", user_key.as_ref(), pool_key_for_pos_mint.as_ref()],
                    ctx.program_id,
                );
                pos_mint = m;
            }
            let position_nft_mint_ai = find_acc(ras, &pos_mint, "position_nft_mint")?;

            // user ATA for position NFT
            let pos_nft_ata_key = anchor_spl::associated_token::get_associated_token_address_with_program_id(
                &user_key,
                &pos_mint,
                &anchor_spl::token::ID,
            );
            let position_nft_account = find_acc(ras, &pos_nft_ata_key, "position_nft_account(user ATA)")?;

            // personal_position（优先已存，否则猜测一个可反序列化的）
            let (personal_position, mut personal_pos_key_maybe) =
                if personal_pos_stored != Pubkey::default() {
                    (find_acc(ras, &personal_pos_stored, "personal_position")?, None)
                } else {
                    let guess_ref = ras
                        .iter()
                        .find(|ai| is_anchor_account::<raydium_amm_v3::states::PersonalPositionState>(ai))
                        .ok_or_else(|| {
                            msg!("missing personal_position in remaining_accounts");
                            error!(OperationError::InvalidParams)
                        })?;
                    let guess = guess_ref.clone();
                    (guess, Some(guess_ref.key()))
                };

            // ---------- parse ZapIn params ----------
            let p: ZapInParams = deserialize_params(&action_bytes)?;
            require!(p.tick_lower < p.tick_upper, OperationError::InvalidTickRange);

            // ca must equal one of the pool mints
            require!(ca_mint == mint0_key || ca_mint == mint1_key, OperationError::InvalidMint);

            // determine which side the deposit came in
            let is_base_input = program_token_mint == mint0_key;

            // ---------- price/fees ----------
            let pool_state_data = raydium_amm_v3::states::PoolState::try_deserialize(&mut &pool_state.try_borrow_data()?[..])
                .map_err(|_| error!(OperationError::InvalidParams))?;
            let sp = pool_state_data.sqrt_price_x64;

            let sp_u = U256::from(sp);
            let q64_u = U256::from(Q64_U128);
            let price_q64 = sp_u.mul_div_floor(sp_u, q64_u).ok_or(error!(OperationError::InvalidParams))?;

            // amm_config fees
            let cfg = raydium_amm_v3::states::AmmConfig::try_deserialize(&mut &amm_config.try_borrow_data()?[..])
                .map_err(|_| error!(OperationError::InvalidParams))?;
            let trade_fee_bps: u32 = cfg.trade_fee_rate.into();
            let protocol_fee_bps: u32 = cfg.protocol_fee_rate.into();
            let total_fee_bps = trade_fee_bps + protocol_fee_bps;

            let slip_bps = p.slippage_bps.min(10_000);
            let one = U256::from(10_000u32);
            let fee_factor = one - U256::from(total_fee_bps);
            let slip_factor = one - U256::from(slip_bps);
            let discount = fee_factor.mul_div_floor(slip_factor, one).ok_or(error!(OperationError::InvalidParams))?;

            let amount_in_u = U256::from(p.amount_in);
            let min_amount_out_u = if is_base_input {
                amount_in_u.mul_div_floor(price_q64, q64_u).ok_or(error!(OperationError::InvalidParams))?
                    .mul_div_floor(discount, one).ok_or(error!(OperationError::InvalidParams))?
            } else {
                amount_in_u.mul_div_floor(q64_u, price_q64.max(U256::from(1u8))).ok_or(error!(OperationError::InvalidParams))?
                    .mul_div_floor(discount, one).ok_or(error!(OperationError::InvalidParams))?
            };
            let min_amount_out = min_amount_out_u.to_underflow_u64();

            // tick-derived checks / tick array consistency
            let sa = tick_math::get_sqrt_price_at_tick(p.tick_lower).map_err(|_| error!(OperationError::InvalidParams))?;
            let sb = tick_math::get_sqrt_price_at_tick(p.tick_upper).map_err(|_| error!(OperationError::InvalidParams))?;
            require!(sa < sb, OperationError::InvalidTickRange);
            require!(sp >= sa && sp <= sb, OperationError::InvalidParams);

            // ---------- refund path when amount < requested ----------
            if amount_total < p.amount_in {
                let refund_cpi = Transfer {
                    from: program_token_account.clone(),
                    to: recipient_refund_ata.clone(),
                    authority: operation_ai.clone(),
                };
                token::transfer(
                    CpiContext::new_with_signer(token_prog_ai.clone(), refund_cpi, signer_seeds),
                    amount_total,
                )?;
                ctx.accounts.operation_data.executed = true;
                msg!("ZapIn refund: expected {}, received {}, refunded all.", p.amount_in, amount_total);
                return Ok(());
            }

            // ---------- compute single-swap split ----------
            let sa_u = U256::from(sa);
            let sb_u = U256::from(sb);
            let sp_u2 = U256::from(sp);
            let sp_minus_sa = if sp_u2 >= sa_u { sp_u2 - sa_u } else { return err!(OperationError::InvalidParams); };
            let sb_minus_sp = if sb_u >= sp_u2 { sb_u - sp_u2 } else { return err!(OperationError::InvalidParams); };
            let r_num = sb_u * sp_minus_sa;
            let r_den = sp_u2 * sb_minus_sp;
            let frac_den = r_den + r_num;
            require!(frac_den > U256::from(0u8), OperationError::InvalidParams);

            let swap_amount = if is_base_input {
                U256::from(p.amount_in).mul_div_floor(r_num, frac_den).ok_or(error!(OperationError::InvalidParams))?
            } else {
                U256::from(p.amount_in).mul_div_floor(r_den, frac_den).ok_or(error!(OperationError::InvalidParams))?
            }.to_underflow_u64();

            // ---------- move deposit to proper PDA input/output account ----------
            let to_acc = if is_base_input { pda_input_ata.clone() } else { pda_output_ata.clone() };
            let move_cpi = Transfer {
                from: program_token_account.clone(),
                to: to_acc,
                authority: operation_ai.clone(),
            };
            token::transfer(
                CpiContext::new_with_signer(token_prog_ai.clone(), move_cpi, signer_seeds),
                amount_total,
            )?;

            // record balances
            let pre_out = if is_base_input { load_token_amount(&pda_output_ata)? } else { load_token_amount(&pda_input_ata)? };
            let pre_in = if is_base_input { load_token_amount(&pda_input_ata)? } else { load_token_amount(&pda_output_ata)? };

            // ---------- swap in pool ----------
            {
                let (in_acc, out_acc, in_vault, out_vault, in_mint, out_mint) = if is_base_input {
                    (pda_input_ata.clone(), pda_output_ata.clone(), vault0.clone(), vault1.clone(), mint0.clone(), mint1.clone())
                } else {
                    (pda_output_ata.clone(), pda_input_ata.clone(), vault1.clone(), vault0.clone(), mint1.clone(), mint0.clone())
                };

                let swap_accounts = cpi::accounts::SwapSingleV2 {
                    payer: operation_ai.clone(),
                    amm_config: amm_config.clone(),
                    pool_state: pool_state.clone(),
                    input_token_account: in_acc,
                    output_token_account: out_acc,
                    input_vault: in_vault,
                    output_vault: out_vault,
                    observation_state: observation.clone(),
                    token_program: token_prog_ai.clone(),
                    token_program_2022: token22_prog_ai.clone(),
                    memo_program: memo_prog_ai.clone(),
                    input_vault_mint: in_mint,
                    output_vault_mint: out_mint,
                };
                let swap_ctx = CpiContext::new(clmm_prog_ai.clone(), swap_accounts)
                    .with_signer(signer_seeds);
                cpi::swap_v2(
                    swap_ctx,
                    swap_amount,
                    min_amount_out,
                    0,
                    is_base_input,
                )?;
            }

            // delta after swap
            let post_out = if is_base_input { load_token_amount(&pda_output_ata)? } else { load_token_amount(&pda_input_ata)? };
            let post_in = if is_base_input { load_token_amount(&pda_input_ata)? } else { load_token_amount(&pda_output_ata)? };
            let received = post_out.checked_sub(pre_out).ok_or(error!(OperationError::InvalidParams))?;
            let spent = pre_in.checked_sub(pre_in.min(post_in)).ok_or(error!(OperationError::InvalidParams))?;
            let remaining = amount_total.checked_sub(spent).ok_or(error!(OperationError::InvalidParams))?;

            // ---------- ensure position_nft_mint account exists (create if empty) ----------
            if position_nft_mint_ai.data_is_empty() {
                let mint_space = spl_token::state::Mint::LEN;
                let rent_lamports = Rent::get()?.minimum_balance(mint_space);

                let create_ix = system_instruction::create_account(
                    &user_key,
                    &pos_mint,
                    rent_lamports,
                    mint_space as u64,
                    &anchor_spl::token::ID,
                );

                // 为 pos_nft_mint PDA 计算 bump
                let (_pk, bump) = Pubkey::find_program_address(
                    &[b"pos_nft_mint", user_key.as_ref(), pool_key_for_pos_mint.as_ref()],
                    ctx.program_id,
                );
                let bump_bytes = [bump];

                let seeds: &[&[u8]] = &[
                    b"pos_nft_mint",
                    user_key.as_ref(),
                    pool_key_for_pos_mint.as_ref(),
                    &bump_bytes,
                ];
                invoke_signed(
                    &create_ix,
                    &[
                        user_ai.clone(),
                        position_nft_mint_ai.clone(),
                        system_prog_ai.clone(),
                    ],
                    &[seeds],
                )?;
            }

            // ---------- open position (mint NFT) ----------
            {
                let open_accounts = cpi::accounts::OpenPositionV2 {
                    payer: operation_ai.clone(),
                    pool_state: pool_state.clone(),
                    position_nft_owner: user_ai.clone(),
                    position_nft_mint: position_nft_mint_ai.clone(),
                    position_nft_account: position_nft_account.clone(),
                    personal_position: personal_position.clone(),
                    protocol_position: protocol_pos.clone(),
                    tick_array_lower: ta_lower.clone(),
                    tick_array_upper: ta_upper.clone(),
                    token_program: token_prog_ai.clone(),
                    system_program: system_prog_ai.clone(),
                    rent: rent_sysvar_ai.clone(),
                    associated_token_program: find_acc(ras, &anchor_spl::associated_token::ID, "associated_token_program")?,
                    token_account_0: pda_input_ata.clone(),
                    token_account_1: pda_output_ata.clone(),
                    token_vault_0: vault0.clone(),
                    token_vault_1: vault1.clone(),
                    vault_0_mint: mint0.clone(),
                    vault_1_mint: mint1.clone(),
                    metadata_program: memo_prog_ai.clone(),
                    metadata_account: position_nft_account.clone(),
                    token_program_2022: token22_prog_ai.clone(),
                };

                let pool = pool_state_data;
                let tick_spacing: i32 = pool.tick_spacing.into();
                let lower_start = tick_array_start_index(p.tick_lower, tick_spacing);
                let upper_start = tick_array_start_index(p.tick_upper, tick_spacing);

                let open_ctx = CpiContext::new(clmm_prog_ai.clone(), open_accounts)
                    .with_signer(signer_seeds);

                cpi::open_position_v2(
                    open_ctx,
                    p.tick_lower,
                    p.tick_upper,
                    lower_start,
                    upper_start,
                    0u128,
                    0u64,
                    0u64,
                    false,            // with_metadata
                    Some(true),       // base_flag
                )?;
            }

            // ---------- increase liquidity (use remaining + received) ----------
            {
                let (amount_0_max, amount_1_max) = if is_base_input { (remaining, received) } else { (received, remaining) };

                let inc_accounts = cpi::accounts::IncreaseLiquidityV2 {
                    nft_owner: user_ai.clone(),
                    nft_account: position_nft_account.clone(),
                    pool_state: pool_state.clone(),
                    protocol_position: protocol_pos.clone(),
                    personal_position: personal_position.clone(),
                    tick_array_lower: ta_lower.clone(),
                    tick_array_upper: ta_upper.clone(),
                    token_account_0: pda_input_ata.clone(),
                    token_account_1: pda_output_ata.clone(),
                    token_vault_0: vault0.clone(),
                    token_vault_1: vault1.clone(),
                    token_program: token_prog_ai.clone(),
                    token_program_2022: token22_prog_ai.clone(),
                    vault_0_mint: mint0.clone(),
                    vault_1_mint: mint1.clone(),
                };
                let inc_ctx = CpiContext::new(clmm_prog_ai, inc_accounts)
                    .with_signer(signer_seeds);
                cpi::increase_liquidity_v2(
                    inc_ctx,
                    0,
                    amount_0_max,
                    amount_1_max,
                    Some(is_base_input),
                )?;
            }
        }

        // 写回（仅在需要时）
        {
            let od_mut = &mut ctx.accounts.operation_data;
            if od_mut.personal_position == Pubkey::default() {
                if let Some(k) = personal_pos_key_maybe.take() {
                    od_mut.personal_position = k;
                }
            }
            if od_mut.position_nft_mint == Pubkey::default() {
                od_mut.position_nft_mint = pos_mint;
            }
            od_mut.executed = true;
        }

        Ok(())
    }

    pub fn claim(ctx: Context<Claim>, transfer_id: String, p: ClaimParams) -> Result<()> {
        // ---- 先拷 key，立刻结束对 od 的借用 ----
        let (operation_key, pool_state_key, amm_config_key, observation_key,
            token_vault_0_key, token_vault_1_key, token_mint_0_key, token_mint_1_key,
            tick_array_lower_key, tick_array_upper_key, protocol_position_key, personal_position_key,
            position_nft_mint_key_opt)
            = {
            let od = &ctx.accounts.operation_data;
            require!(od.initialized, OperationError::NotInitialized);
            require!(od.transfer_id == transfer_id, OperationError::InvalidTransferId);
            (
                od.key(),
                od.pool_state, od.amm_config, od.observation_state,
                od.token_vault_0, od.token_vault_1, od.token_mint_0, od.token_mint_1,
                od.tick_array_lower, od.tick_array_upper, od.protocol_position, od.personal_position,
                if od.position_nft_mint != Pubkey::default() { Some(od.position_nft_mint) } else { None },
            )
        };
        let user_key = ctx.accounts.user.key();
        let od_key   = ctx.accounts.operation_data.key();
        let clmm_pid = { let od = &ctx.accounts.operation_data; od.clmm_program_id };

        // 不复制 remaining_accounts，直接借用
        let ras = ctx.remaining_accounts;

        let token_prog_ai   = find_acc(ras, &token::ID, "token_program")?;
        let clmm_prog_ai    = find_acc(ras, &clmm_pid, "clmm_program")?;
        let token22_prog_ai = find_acc(ras, &Token2022::id(), "token_program_2022")?;
        let memo_prog_ai    = find_acc(ras, &spl_memo::id(), "memo_program")?;
        let user_ai         = find_acc(ras, &user_key, "user")?;


        // ---- 从 remaining_accounts 抓取所有 Raydium / Vault / Mint / Position 账户 ----
        let pool_state          = find_acc(ras, &pool_state_key,        "pool_state")?;
        let amm_config          = find_acc(ras, &amm_config_key,        "amm_config")?;
        let observation_state   = find_acc(ras, &observation_key,       "observation_state")?;
        let token_vault_0       = find_acc(ras, &token_vault_0_key,     "token_vault_0")?;
        let token_vault_1       = find_acc(ras, &token_vault_1_key,     "token_vault_1")?;
        let token_mint_0        = find_acc(ras, &token_mint_0_key,      "token_mint_0")?;
        let token_mint_1        = find_acc(ras, &token_mint_1_key,      "token_mint_1")?;
        let tick_array_lower_ai = find_acc(ras, &tick_array_lower_key,  "tick_array_lower")?;
        let tick_array_upper_ai = find_acc(ras, &tick_array_upper_key,  "tick_array_upper")?;
        let protocol_position   = find_acc(ras, &protocol_position_key, "protocol_position")?;
        let personal_position   = find_acc(ras, &personal_position_key, "personal_position")?;
        let operation_ai = find_acc(ras, &od_key, "operation_data_pda")?;

        let input_token_account  = find_pda_token_by_mint(ras, &operation_key, &token_mint_0_key, "input_token_account")?;
        let output_token_account = find_pda_token_by_mint(ras, &operation_key, &token_mint_1_key, "output_token_account")?;

        // ---- 计算/获取 position NFT mint & ATA ----
        let pos_mint = position_nft_mint_key_opt.unwrap_or_else(|| {
            let (m, _) = Pubkey::find_program_address(
                &[b"pos_nft_mint", user_key.as_ref(), pool_state_key.as_ref()],
                ctx.program_id,
            );
            m
        });
        let position_nft_account_key = get_associated_token_address_with_program_id(
            &user_key, &pos_mint, &anchor_spl::token::ID,
        );
        let position_nft_account_ai = find_acc(ras, &position_nft_account_key, "position_nft_account(user ATA of position NFT)")?;

        // ---- NFT 归属校验 ----
        {
            let nft_acc = spl_token::state::Account::unpack(&position_nft_account_ai.try_borrow_data()?)
                .map_err(|_| error!(OperationError::InvalidParams))?;
            require!(nft_acc.owner == user_key, OperationError::Unauthorized);
            require!(nft_acc.mint == pos_mint,                  OperationError::Unauthorized);
            require!(nft_acc.amount == 1,                       OperationError::Unauthorized);
        }

        // ---- 领取 USDC 的 ATA（从 remaining_accounts 里）----
        let recipient_token_account =
            find_user_token_by_mint(ras, &user_key, &token_mint_0_key, "recipient_token_account(token_mint_0)")
                .or_else(|_| find_user_token_by_mint(ras, &user_key, &token_mint_1_key, "recipient_token_account(token_mint_1)"))?;
        let usdc_mint = {
            let acc = spl_token::state::Account::unpack(&recipient_token_account.try_borrow_data()?)
                .map_err(|_| error!(OperationError::InvalidParams))?;
            require!(acc.mint == token_mint_0_key || acc.mint == token_mint_1_key, OperationError::InvalidMint);
            acc.mint
        };

        // —— 记录 claim 前余额（PDA 名下两边）——
        let pre0 = load_token_amount(&input_token_account)?;
        let pre1 = load_token_amount(&output_token_account)?;
        let pre_usdc = if usdc_mint == token_mint_0_key { pre0 } else { pre1 };

        // seeds
        let bump = ctx.bumps.operation_data;
        let h = transfer_id_hash_bytes(&transfer_id);
        let signer_seeds_slice: [&[u8]; 3] = [b"operation_data".as_ref(), h.as_ref(), &[bump]];
        let signer_seeds: &[&[&[u8]]] = &[&signer_seeds_slice];

        // 1) 只结算手续费（liquidity=0）
        {
            let dec_accounts = cpi::accounts::DecreaseLiquidityV2 {
                nft_owner:                 user_ai.clone(),
                nft_account:               position_nft_account_ai.clone(),
                pool_state:                pool_state.clone(),
                protocol_position:         protocol_position.clone(),
                personal_position:         personal_position.clone(),
                tick_array_lower:          tick_array_lower_ai.clone(),
                tick_array_upper:          tick_array_upper_ai.clone(),
                recipient_token_account_0: input_token_account.clone(),
                recipient_token_account_1: output_token_account.clone(),
                token_vault_0:             token_vault_0.clone(),
                token_vault_1:             token_vault_1.clone(),
                token_program:             token_prog_ai.clone(),
                token_program_2022:        token22_prog_ai.clone(),
                vault_0_mint:              token_mint_0.clone(),
                vault_1_mint:              token_mint_1.clone(),
                memo_program:              memo_prog_ai.clone(),
            };
            let dec_ctx = CpiContext::new(clmm_prog_ai.clone(), dec_accounts).with_signer(signer_seeds);
            cpi::decrease_liquidity_v2(dec_ctx, 0u128, 0u64, 0u64)?;
        }

        // 刚领取到 PDA 的手续费数量
        let post0 = load_token_amount(&input_token_account)?;
        let post1 = load_token_amount(&output_token_account)?;
        let got0 = post0.checked_sub(pre0).ok_or(error!(OperationError::InvalidParams))?;
        let got1 = post1.checked_sub(pre1).ok_or(error!(OperationError::InvalidParams))?;
        if got0 == 0 && got1 == 0 {
            msg!("No rewards available to claim right now.");
            return Ok(());
        }

        // 2) 将非 USDC 一侧全量 swap 成 USDC
        let mut total_usdc_after_swap = if usdc_mint == token_mint_0_key { pre_usdc + got0 } else { pre_usdc + got1 };
        if (usdc_mint == token_mint_0_key && got1 > 0) || (usdc_mint == token_mint_1_key && got0 > 0) {
            let (in_acc, out_acc, in_vault, out_vault, in_mint, out_mint, is_base_input, amount_in) =
                if usdc_mint == token_mint_0_key {
                    (output_token_account.clone(), input_token_account.clone(),
                     token_vault_1.clone(), token_vault_0.clone(),
                     token_mint_1.clone(), token_mint_0.clone(),
                     false, got1)
                } else {
                    (input_token_account.clone(), output_token_account.clone(),
                     token_vault_0.clone(), token_vault_1.clone(),
                     token_mint_0.clone(), token_mint_1.clone(),
                     true, got0)
                };


            let swap_accounts = cpi::accounts::SwapSingleV2 {
                payer:                 operation_ai.clone(),
                amm_config:            amm_config.clone(),
                pool_state:            pool_state.clone(),
                input_token_account:   in_acc,
                output_token_account:  out_acc,
                input_vault:           in_vault,
                output_vault:          out_vault,
                observation_state:     observation_state.clone(),
                token_program:         token_prog_ai.clone(),
                token_program_2022:    token22_prog_ai.clone(),
                memo_program:          memo_prog_ai.clone(),
                input_vault_mint:      in_mint,
                output_vault_mint:     out_mint,
            };
            let swap_ctx = CpiContext::new(clmm_prog_ai.clone(), swap_accounts).with_signer(signer_seeds);
            cpi::swap_v2(swap_ctx, amount_in, 0, 0, is_base_input)?;

            // 刷新 USDC 余额
            total_usdc_after_swap = if usdc_mint == token_mint_0_key {
                load_token_amount(&input_token_account)?
            } else {
                load_token_amount(&output_token_account)?
            };
        }

        // 3) 最小到手保护 + 从 PDA 转给 user 的 USDC ATA
        require!(total_usdc_after_swap >= p.min_usdc_out, OperationError::InvalidParams);

        let transfer_from = if usdc_mint == token_mint_0_key {
            input_token_account.clone()
        } else {
            output_token_account.clone()
        };
        let transfer_accounts = Transfer {
            from:      transfer_from,
            to:        recipient_token_account.clone(),
            authority: operation_ai.clone(),
        };
        let token_ctx = CpiContext::new(token_prog_ai.clone(), transfer_accounts).with_signer(signer_seeds);
        token::transfer(token_ctx, total_usdc_after_swap)?;

        emit!(ClaimEvent {
            pool: pool_state_key,
            beneficiary: user_key,
            mint: usdc_mint,
            amount: total_usdc_after_swap,
        });

        Ok(())
    }

    pub fn withdraw(
        ctx: Context<ZapOutExecute>,
        transfer_id: String,
        _bounds: PositionBounds,
        p: ZapOutParams
    ) -> Result<()> {
        let (operation_key, expected_recipient, pool_state_key, amm_config_key, observation_key,
            token_vault_0_key, token_vault_1_key, token_mint_0_key, token_mint_1_key,
            tick_array_lower_key, tick_array_upper_key, protocol_position_key, personal_position_key,
            pos_mint_key_opt)
            = {
            let od = &ctx.accounts.operation_data;

            require!(od.initialized, OperationError::NotInitialized);
            require!(!od.executed, OperationError::AlreadyExecuted);
            require!(od.amount > 0, OperationError::InvalidAmount);

            let expected_recipient = if od.recipient != Pubkey::default() { od.recipient } else { od.authority };

            (
                od.key(),
                expected_recipient,
                od.pool_state, od.amm_config, od.observation_state,
                od.token_vault_0, od.token_vault_1, od.token_mint_0, od.token_mint_1,
                od.tick_array_lower, od.tick_array_upper, od.protocol_position, od.personal_position,
                if od.position_nft_mint != Pubkey::default() { Some(od.position_nft_mint) } else { None },
            )
        };

        let user_key = ctx.accounts.user.key();
        let od_key   = ctx.accounts.operation_data.key();
        let clmm_pid = { let od = &ctx.accounts.operation_data; od.clmm_program_id };
        let this_program_id: Pubkey = *ctx.program_id;

        // seeds
        let bump = ctx.bumps.operation_data;
        let h = transfer_id_hash_bytes(&transfer_id);
        let signer_seeds_slice: [&[u8]; 3] = [b"operation_data".as_ref(), h.as_ref(), &[bump]];
        let signer_seeds: &[&[&[u8]]] = &[&signer_seeds_slice];

        let ras = ctx.remaining_accounts;
        let clmm_prog_ai    = find_acc(ras, &clmm_pid, "clmm_program")?;
        let token_prog_ai   = find_acc(ras, &token::ID, "token_program")?;
        let token22_prog_ai = find_acc(ras, &Token2022::id(), "token_program_2022")?;
        let memo_prog_ai    = find_acc(ras, &spl_memo::id(), "memo_program")?;
        let user_ai         = find_acc(ras, &user_key, "user")?;
        let operation_ai = find_acc(ras, &od_key, "operation_data_pda")?;

        // ---- 从 remaining_accounts 抓所有账户 ----
        let pool_state          = find_acc(ras, &pool_state_key,        "pool_state")?;
        let amm_config          = find_acc(ras, &amm_config_key,        "amm_config")?;
        let observation_state   = find_acc(ras, &observation_key,       "observation_state")?;
        let token_vault_0       = find_acc(ras, &token_vault_0_key,     "token_vault_0")?;
        let token_vault_1       = find_acc(ras, &token_vault_1_key,     "token_vault_1")?;
        let token_mint_0        = find_acc(ras, &token_mint_0_key,      "token_mint_0")?;
        let token_mint_1        = find_acc(ras, &token_mint_1_key,      "token_mint_1")?;
        let tick_array_lower_ai = find_acc(ras, &tick_array_lower_key,  "tick_array_lower")?;
        let tick_array_upper_ai = find_acc(ras, &tick_array_upper_key,  "tick_array_upper")?;
        let protocol_position   = find_acc(ras, &protocol_position_key, "protocol_position")?;
        let personal_position   = find_acc(ras, &personal_position_key, "personal_position")?;

        let input_token_account  = find_pda_token_by_mint(ras, &operation_key, &token_mint_0_key, "input_token_account")?;
        let output_token_account = find_pda_token_by_mint(ras, &operation_key, &token_mint_1_key, "output_token_account")?;
        let recipient_token_account =
            find_user_token_by_mint(ras, &user_key, &token_mint_0_key, "recipient_token_account(token_mint_0)")
                .or_else(|_| find_user_token_by_mint(ras, &user_key, &token_mint_1_key, "recipient_token_account(token_mint_1)"))?;

        // ---- 校验收款人 ATA ----
        let rec_acc = recipient_token_account.clone();
        {
            let ta = spl_token::state::Account::unpack(&rec_acc.try_borrow_data()?)
                .map_err(|_| error!(OperationError::InvalidParams))?;
            require!(ta.owner == expected_recipient, OperationError::Unauthorized);
            let want_mint = if p.want_base { token_mint_0_key } else { token_mint_1_key };
            require!(ta.mint == want_mint, OperationError::InvalidMint);
        }

        // 赎回前余额
        let pre0 = load_token_amount(&input_token_account)?;
        let pre1 = load_token_amount(&output_token_account)?;

        // 读取 position（用于估算最小值）
        let pp_data = personal_position.try_borrow_data()?;
        let pp = raydium_amm_v3::states::PersonalPositionState::try_deserialize(&mut &pp_data[..])
            .map_err(|_| error!(OperationError::InvalidParams))?;
        let full_liquidity: u128 = pp.liquidity;
        require!(full_liquidity > 0, OperationError::InvalidParams);
        let burn_liq: u128 = if p.liquidity_to_burn_u64 > 0 { p.liquidity_to_burn_u64 as u128 } else { full_liquidity };
        require!(burn_liq <= full_liquidity, OperationError::InvalidParams);

        let tick_lower = pp.tick_lower_index;
        let tick_upper = pp.tick_upper_index;

        let sa = tick_math::get_sqrt_price_at_tick(tick_lower).map_err(|_| error!(OperationError::InvalidParams))?;
        let sb = tick_math::get_sqrt_price_at_tick(tick_upper).map_err(|_| error!(OperationError::InvalidParams))?;
        require!(sa < sb, OperationError::InvalidTickRange);

        // 当前价
        let sp = {
            let ps = raydium_amm_v3::states::PoolState::try_deserialize(&mut &pool_state.try_borrow_data()?[..])
                .map_err(|_| error!(OperationError::InvalidParams))?;
            ps.sqrt_price_x64
        };

        // 估算最小到手
        let (est0, est1) = amounts_from_liquidity_burn_q64(sa, sb, sp, burn_liq);
        let min0 = apply_slippage_min(est0, p.slippage_bps);
        let min1 = apply_slippage_min(est1, p.slippage_bps);

        // 预先计算 user 的 position NFT ATA（从 remaining_accounts 找到拥有型）
        let position_nft_account_ai = {
            let pos_mint = pos_mint_key_opt.unwrap_or_else(|| {
                let (m, _) = Pubkey::find_program_address(
                    &[b"pos_nft_mint", user_key.as_ref(), pool_state_key.as_ref()],
                    &this_program_id,
                );
                m
            });
            let ata = get_associated_token_address_with_program_id(
                &user_key,
                &pos_mint,
                &anchor_spl::token::ID,
            );
            find_acc(ras, &ata, "position_nft_account")?
        };

        // ---------- A: 赎回 ----------
        {
            let dec_accounts = cpi::accounts::DecreaseLiquidityV2 {
                nft_owner:                 user_ai.clone(),
                nft_account:               position_nft_account_ai.clone(),
                pool_state:                pool_state.clone(),
                protocol_position:         protocol_position.clone(),
                personal_position:         personal_position.clone(),
                tick_array_lower:          tick_array_lower_ai.clone(),
                tick_array_upper:          tick_array_upper_ai.clone(),
                recipient_token_account_0: input_token_account.clone(),
                recipient_token_account_1: output_token_account.clone(),
                token_vault_0:             token_vault_0.clone(),
                token_vault_1:             token_vault_1.clone(),
                token_program:             token_prog_ai.clone(),
                token_program_2022:        token22_prog_ai.clone(),
                vault_0_mint:              token_mint_0.clone(),
                vault_1_mint:              token_mint_1.clone(),
                memo_program:              memo_prog_ai.clone(),
            };
            let dec_ctx = CpiContext::new(clmm_prog_ai.clone(), dec_accounts)
                .with_signer(signer_seeds);
            cpi::decrease_liquidity_v2(dec_ctx, burn_liq, min0, min1)?;
        }

        // 赎回后增量
        let post0 = load_token_amount(&input_token_account)?;
        let post1 = load_token_amount(&output_token_account)?;
        let got0  = post0.checked_sub(pre0).ok_or(error!(OperationError::InvalidParams))?;
        let got1  = post1.checked_sub(pre1).ok_or(error!(OperationError::InvalidParams))?;

        // ---------- B: 单边换（可选） ----------
        let (mut total_out, swap_amount, is_base_input) = if p.want_base {
            (got0, got1, false)
        } else {
            (got1, got0, true)
        };

        if swap_amount > 0 {
            let (in_acc, out_acc, in_vault, out_vault, in_mint, out_mint) =
                if p.want_base {
                    (output_token_account.clone(), input_token_account.clone(),
                     token_vault_1.clone(), token_vault_0.clone(),
                     token_mint_1.clone(), token_mint_0.clone())
                } else {
                    (input_token_account.clone(), output_token_account.clone(),
                     token_vault_0.clone(), token_vault_1.clone(),
                     token_mint_0.clone(), token_mint_1.clone())
                };

            {
                let swap_accounts = cpi::accounts::SwapSingleV2 {
                    payer:                 operation_ai.clone(),
                    amm_config:            amm_config.clone(),
                    pool_state:            pool_state.clone(),
                    input_token_account:   in_acc,
                    output_token_account:  out_acc,
                    input_vault:           in_vault,
                    output_vault:          out_vault,
                    observation_state:     observation_state.clone(),
                    token_program:         token_prog_ai.clone(),
                    token_program_2022:    token22_prog_ai.clone(),
                    memo_program:          memo_prog_ai.clone(),
                    input_vault_mint:      in_mint,
                    output_vault_mint:     out_mint,
                };
                let swap_ctx = CpiContext::new(clmm_prog_ai.clone(), swap_accounts)
                    .with_signer(signer_seeds);
                cpi::swap_v2(swap_ctx, swap_amount, 0, 0, is_base_input)?;

            }
            // 刷新单边后的总量
            total_out = if p.want_base {
                load_token_amount(&input_token_account)?
                    .checked_sub(pre0).ok_or(error!(OperationError::InvalidParams))?
            } else {
                load_token_amount(&output_token_account)?
                    .checked_sub(pre1).ok_or(error!(OperationError::InvalidParams))?
            };
        }

        // ---------- C: 最低到手 ----------
        require!(total_out >= p.min_payout, OperationError::InvalidParams);

        // ---------- D: 打款给收款人 ----------
        let from_acc = if p.want_base { input_token_account.clone() } else { output_token_account.clone() };
        let cpi_accounts = Transfer {
            from:      from_acc,
            to:        rec_acc.clone(),
            authority: operation_ai.clone(),
        };
        token::transfer(
            CpiContext::new_with_signer(token_prog_ai.clone(), cpi_accounts, signer_seeds),
            total_out,
        )?;

        // 标记执行完毕
        ctx.accounts.operation_data.executed = true;
        Ok(())
    }
    // Modify PDA Authority
    pub fn modify_pda_authority(
        ctx: Context<ModifyPdaAuthority>,
        new_authority: Pubkey,
    ) -> Result<()> {
        let operation_data = &mut ctx.accounts.operation_data;

        // Verify current authority
        require!(operation_data.initialized, OperationError::NotInitialized);
        require!(
            operation_data.authority == ctx.accounts.current_authority.key(),
            OperationError::Unauthorized
        );

        // Update authority
        operation_data.authority = new_authority;
        msg!("Update PDA Authority to: {}", new_authority);
        Ok(())
    }
}

fn find_acc<'info>(ras: &'info [AccountInfo<'info>], key: &Pubkey, label: &str) -> Result<AccountInfo<'info>> {
    let ai = ras.iter().find(|ai| ai.key == key).ok_or_else(|| {
        msg!("missing account in remaining_accounts: {} = {}", label, key);
        error!(OperationError::InvalidParams)
    })?;
    Ok(ai.clone()) //
}

fn unpack_token_account(ai: &AccountInfo) -> Option<spl_token::state::Account> {
    let data = ai.try_borrow_data().ok()?;
    spl_token::state::Account::unpack_from_slice(&data).ok()
}
fn find_pda_token_by_mint<'info>(
    ras: &'info [AccountInfo<'info>],
    owner: &Pubkey,
    mint: &Pubkey,
    label: &str,
) -> Result<AccountInfo<'info>> {
    for ai in ras {
        if let Some(ta) = unpack_token_account(ai) {
            if ta.owner == *owner && ta.mint == *mint {
                return Ok(ai.clone()); // 返回 owned AccountInfo
            }
        }
    }
    msg!("missing account in remaining_accounts: {} (owner={}, mint={})", label, owner, mint);
    err!(OperationError::InvalidParams)
}

fn find_user_token_by_mint<'info>(
    ras: &'info [AccountInfo<'info>],
    user: &Pubkey,
    mint: &Pubkey,
    label: &str,
) -> Result<AccountInfo<'info>> {
    for ai in ras {
        if let Some(ta) = unpack_token_account(ai) {
            if ta.owner == *user && ta.mint == *mint {
                return Ok(ai.clone()); // 返回 owned AccountInfo
            }
        }
    }
    msg!("missing account in remaining_accounts: {} (owner=user {}, mint={})", label, user, mint);
    err!(OperationError::InvalidParams)
}

fn try_deser_anchor_account<T: AccountDeserialize>(ai: &AccountInfo) -> Option<T> {
    let data_ref = ai.try_borrow_data().ok()?;   // Ref<[u8]>
    let mut bytes: &[u8] = &data_ref;            // &Ref<[u8]> -> &[u8]
    T::try_deserialize(&mut bytes).ok()
}

/// 只检查某 AccountInfo 是否是某个 Anchor 类型（通过 try_deserialize 是否成功）
fn is_anchor_account<T: AccountDeserialize>(ai: &AccountInfo) -> bool {
    try_deser_anchor_account::<T>(ai).is_some()
}

fn load_token_amount(ai: &AccountInfo) -> Result<u64> {
    let data = ai.try_borrow_data()?;
    let acc = spl_token::state::Account::unpack_from_slice(&data)
        .map_err(|_| error!(OperationError::InvalidParams))?;
    Ok(acc.amount)
}

const Q64_U128: u128 = 1u128 << 64;

#[inline]
fn transfer_id_hash_bytes(transfer_id: &str) -> [u8; 32] {
    solana_hash(transfer_id.as_bytes()).to_bytes()
}
#[inline]
fn amounts_from_liquidity_burn_q64(
    sa: u128,    // sqrt(P_lower) in Q64.64
    sb: u128,    // sqrt(P_upper) in Q64.64
    sp: u128,    // sqrt(P_current) in Q64.64
    d_liq: u128, // ΔL (liquidity to burn)
) -> (u64, u64) {
    assert!(sa < sb, "invalid tick bounds");
    if d_liq == 0 {
        return (0, 0);
    }
    let sa_u = U256::from(sa);
    let sb_u = U256::from(sb);
    let sp_u = U256::from(sp);
    let dL_u = U256::from(d_liq);
    let q64  = U256::from(Q64_U128);
    let diff_sb_sa = sb_u - sa_u;

    let (amount0_u256, amount1_u256) = if sp_u <= sa_u {
        let num0 = dL_u * diff_sb_sa * q64;
        let den0 = sa_u * sb_u;
        let a0 = num0.mul_div_floor(U256::from(1u8), den0).unwrap_or(U256::from(0));
        (a0, U256::from(0))
    } else if sp_u >= sb_u {
        let a1 = (dL_u * diff_sb_sa).mul_div_floor(U256::from(1u8), q64).unwrap_or(U256::from(0));
        (U256::from(0), a1)
    } else {
        let num0 = dL_u * (sb_u - sp_u) * q64;
        let den0 = sp_u * sb_u;
        let a0 = num0.mul_div_floor(U256::from(1u8), den0).unwrap_or(U256::from(0));
        let a1 = (dL_u * (sp_u - sa_u)).mul_div_floor(U256::from(1u8), q64).unwrap_or(U256::from(0));
        (a0, a1)
    };

    let amount0 = amount0_u256.to_underflow_u64();
    let amount1 = amount1_u256.to_underflow_u64();
    (amount0, amount1)
}

const TICK_ARRAY_SIZE: i32 = 88; //Raydium/UniV3 每个 TickArray 覆盖 88 个 tick 间隔
#[inline]
fn tick_array_start_index(tick_index: i32, tick_spacing: i32) -> i32 {
    let span = tick_spacing * TICK_ARRAY_SIZE;
    // floor 除法，处理负 tick
    let q = if tick_index >= 0 {
        tick_index / span
    } else {
        (tick_index - (span - 1)) / span
    };
    q * span
}

/// Helper function to deserialize params
fn deserialize_params<T: AnchorDeserialize>(data: &[u8]) -> Result<T> {
    T::try_from_slice(data).map_err(|_| error!(OperationError::InvalidParams))
}

#[inline]
fn apply_slippage_min(amount: u64, slippage_bps: u32) -> u64 {
    // min_out = amount * (1 - bps/1e4)
    let one = 10_000u128;
    let bps = (slippage_bps as u128).min(one);
    let num = (amount as u128).saturating_mul(one.saturating_sub(bps));
    (num / one) as u64
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + OperationData::LEN,
        seeds = [b"operation_data"],
        bump
    )]
    pub operation_data: Account<'info, OperationData>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(transfer_id: String)]
pub struct Deposit<'info> {
    #[account(
        init_if_needed,
        payer = authority,
        space = 8 + OperationData::LEN,
        seeds = [
        b"operation_data".as_ref(),
        transfer_id_hash_bytes(&transfer_id).as_ref(),
        ],
        bump
    )]
    pub operation_data: Account<'info, OperationData>,

    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        mut,
        constraint = authority_ata.owner == authority.key() @ OperationError::Unauthorized
    )]
    pub authority_ata: Account<'info, TokenAccount>,

    #[account(
        mut,
        constraint = program_token_account.owner == operation_data.key() @ OperationError::InvalidProgramAccount
    )]
    pub program_token_account: Account<'info, TokenAccount>,

    // ===== 新增：Raydium CPI 相关账户（只读一致性，deposit 时校验并落库） =====
    #[account(constraint = clmm_program.key() == RAYDIUM_CLMM_PROGRAM_ID)]
    pub clmm_program: Program<'info, AmmV3>,

    // 池 & 配置
    #[account(mut)]
    pub pool_state: AccountLoader<'info, PoolState>,
    #[account(address = pool_state.load()?.amm_config)]
    pub amm_config: Box<Account<'info, AmmConfig>>,
    #[account(mut, address = pool_state.load()?.observation_key)]
    pub observation_state: AccountLoader<'info, ObservationState>,

    // Vault & Mint
    #[account(mut, address = pool_state.load()?.token_vault_0)]
    pub token_vault_0: Box<InterfaceAccount<'info, InterfaceTokenAccount>>,
    #[account(mut, address = pool_state.load()?.token_vault_1)]
    pub token_vault_1: Box<InterfaceAccount<'info, InterfaceTokenAccount>>,
    #[account(address = pool_state.load()?.token_mint_0)]
    pub token_mint_0: Box<InterfaceAccount<'info, InterfaceMint>>,
    #[account(address = pool_state.load()?.token_mint_1)]
    pub token_mint_1: Box<InterfaceAccount<'info, InterfaceMint>>,

    // 系统/程序
    #[account(constraint = token_program.key() == token::ID @ OperationError::InvalidTokenProgram)]
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct PositionBounds {
    pub tick_lower: i32,
    pub tick_upper: i32,
}

#[derive(Accounts)]
#[instruction(transfer_id: String)]
pub struct Claim<'info> {
    // ZapIn 的 PDA（vault authority），用 transfer_id 维度
    #[account(
        mut,
        seeds = [b"operation_data".as_ref(), transfer_id_hash_bytes(&transfer_id).as_ref()],
        bump
    )]
    pub operation_data: Account<'info, OperationData>,

    // 只有 user（签名者）才能 claim
    pub user: Signer<'info>,

    /// CHECK: spl-memo
    #[account(address = spl_memo::id())]
    pub memo_program: UncheckedAccount<'info>,

    // 程序
    #[account(constraint = clmm_program.key() == RAYDIUM_CLMM_PROGRAM_ID)]
    pub clmm_program: Program<'info, AmmV3>,
    pub token_program: Program<'info, Token>,
    pub token_program_2022: Program<'info, Token2022>,
}

#[event]
pub struct ClaimEvent {
    pub pool: Pubkey,
    pub beneficiary: Pubkey, // = user_da
    pub mint: Pubkey,        // 实际 USDC mint
    pub amount: u64,         // 实转金额
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct ClaimParams {
    /// 领取后，最终到手的 USDC 不得低于该值
    pub min_usdc_out: u64, // required
}

#[derive(Accounts)]
#[instruction(transfer_id: String)]
pub struct Execute<'info> {
    #[account(
        mut,
        seeds = [
        b"operation_data".as_ref(),
        transfer_id_hash_bytes(&transfer_id).as_ref(),
        ],
        bump
    )]
    pub operation_data: Box<Account<'info, OperationData>>,

    // 用户作为 position NFT 的所有者和 payer
    #[account(mut)]
    pub user: Signer<'info>,

    // 程序/系统
    #[account(address = spl_memo::id())]
    pub memo_program: UncheckedAccount<'info>,
    #[account(constraint = clmm_program.key() == operation_data.clmm_program_id)]
    pub clmm_program: Program<'info, AmmV3>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub token_program: Program<'info, Token>,
    pub token_program_2022: Program<'info, Token2022>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
#[instruction(transfer_id: String, bounds: PositionBounds)]
pub struct ZapOutExecute<'info> {
    // 程序 PDA（vault authority），transfer_id 维度
    #[account(
        mut,
        seeds = [
        b"operation_data".as_ref(),
        transfer_id_hash_bytes(&transfer_id).as_ref(),
        ],
        bump
    )]
    pub operation_data: Box<Account<'info, OperationData>>,

    // ====== 接收账户（实际打款目标），mint 运行时校验 ======
    #[account(mut)]
    pub recipient_token_account: Box<InterfaceAccount<'info, InterfaceTokenAccount>>,

    // ====== Position / Pool / Raydium 帐户 ======
    /// CHECK: 仅作转发给 Raydium 的 nft_owner（不要求签名）
    pub user: UncheckedAccount<'info>,

    /// CHECK: spl-memo
    #[account(address = spl_memo::id())]
    pub memo_program: UncheckedAccount<'info>,

    // 程序
    #[account(constraint = clmm_program.key() == RAYDIUM_CLMM_PROGRAM_ID)]
    pub clmm_program: Program<'info, AmmV3>,
    pub token_program: Program<'info, Token>,
    pub token_program_2022: Program<'info, Token2022>,
    pub system_program: Program<'info, System>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct ZapOutParams {
    /// 期望拿回哪一侧：true=base(token0)，false=quote(token1)
    pub want_base: bool,
    /// 允许的滑点（bps）
    pub slippage_bps: u32,
    /// 要赎回的流动性（为 0 时表示全仓位）
    pub liquidity_to_burn_u64: u64,
    /// 整体流程最终至少要拿回的目标侧资产数量
    pub min_payout: u64,
}
#[derive(Accounts)]
pub struct ModifyPdaAuthority<'info> {
    #[account(
        mut,
        seeds = [b"operation_data"],
        bump
    )]
    pub operation_data: Account<'info, OperationData>,
    #[account(
        constraint = current_authority.key() == operation_data.authority @ OperationError::Unauthorized
    )]
    pub current_authority: Signer<'info>,
}

#[account]
#[derive(Default)]
pub struct OperationData {
    pub authority: Pubkey,
    pub initialized: bool,
    pub transfer_id: String,
    pub recipient: Pubkey,
    pub operation_type: OperationType,
    pub action: Vec<u8>, // Serialize operation-specific parameters
    pub amount: u64,
    pub executed: bool,
    pub ca: Pubkey, // contract address

    // ===== Raydium CLMM & 池静态信息（deposit 时落库） =====
    pub clmm_program_id: Pubkey,   // 冗余存储，便于 seeds::program 校验
    pub pool_state: Pubkey,
    pub amm_config: Pubkey,
    pub observation_state: Pubkey,
    pub token_vault_0: Pubkey,
    pub token_vault_1: Pubkey,
    pub token_mint_0: Pubkey,
    pub token_mint_1: Pubkey,

    // ===== ZapIn/Position 相关 =====
    pub tick_lower: i32,
    pub tick_upper: i32,
    pub tick_array_lower: Pubkey,
    pub tick_array_upper: Pubkey,
    pub protocol_position: Pubkey,
    pub personal_position: Pubkey,
    pub position_nft_mint: Pubkey,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq, Debug)]
pub enum OperationType {
    Transfer,
    ZapIn,
}

impl Default for OperationType {
    fn default() -> Self {
        OperationType::Transfer
    }
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct TransferParams {
    pub amount: u64,
    pub recipient: Pubkey,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct ZapInParams {
    pub amount_in: u64, // required
    pub pool: Pubkey, // required
    pub tick_lower: i32, // required
    pub tick_upper: i32, // required
    pub slippage_bps: u32, // required
}

impl OperationData {
    pub const LEN: usize =
        32 + 1 + (4 + 64) + 32 + 1 + (4 + 256) + 8 + 1 + 32
            // 新字段（Raydium 固定 8 * Pubkey + ticks + 3 * Pubkey）
            + 32 // clmm_program_id
            + 32 // pool_state
            + 32 // amm_config
            + 32 // observation_state
            + 32 // token_vault_0
            + 32 // token_vault_1
            + 32 // token_mint_0
            + 32 // token_mint_1
            + 4  // tick_lower (i32)
            + 4  // tick_upper (i32)
            + 32 // tick_array_lower
            + 32 // tick_array_upper
            + 32 // protocol_position
            + 32 // personal_position
            + 32 ; // position_nft_mint

}

#[error_code]
pub enum OperationError {
    #[msg("PDA not initialized")]
    NotInitialized,
    #[msg("Invalid transfer amount")]
    InvalidAmount,
    #[msg("Invalid transfer ID")]
    InvalidTransferId,
    #[msg("Transfer already executed")]
    AlreadyExecuted,
    #[msg("Unauthorized access")]
    Unauthorized,
    #[msg("Invalid mint")]
    InvalidMint,
    #[msg("Invalid token program")]
    InvalidTokenProgram,
    #[msg("Invalid parameters")]
    InvalidParams,
    #[msg("Invalid tick range")]
    InvalidTickRange,
    #[msg("Invalid program account")]
    InvalidProgramAccount,
}