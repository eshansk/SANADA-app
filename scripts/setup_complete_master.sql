-- =============================================================================
-- SANADA Complete Master Database Schema & Functions
-- =============================================================================
-- Paste and run this ENTIRE script in your Supabase SQL Editor.
-- It is 100% idempotent and safe to re-run anytime:
--  - Does NOT delete or reset any existing users, passwords, API keys, or runs.
--  - Combines Users + Auth + API Keys + Run History + Password Change + Uniqueness.
--  - Fixes candidate function ambiguity for change_password().
--  - Drops obsolete auth triggers that cause "Database error saving new user".
--  - Enforces strict case-insensitive uniqueness on usernames and emails.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 0. Drop Obsolete / Broken Triggers on auth.users
-- (Fixes "Database error saving new user" during Supabase signUp)
-- -----------------------------------------------------------------------------
drop trigger if exists on_auth_user_created on auth.users;
drop function if exists public.handle_new_user() cascade;
drop function if exists public.handle_new_auth_user() cascade;

-- -----------------------------------------------------------------------------
-- 1. Extensions
-- -----------------------------------------------------------------------------
create extension if not exists pgcrypto;

-- -----------------------------------------------------------------------------
-- 2. Table: public.app_users & Unique Constraints
-- -----------------------------------------------------------------------------
create table if not exists public.app_users (
    id uuid primary key default gen_random_uuid(),
    username text not null,
    password text not null,
    role text check (role in ('developer', 'user')) not null default 'user',
    email text,
    created_at timestamptz not null default now()
);

-- Ensure email column exists if table was originally created without it
alter table public.app_users add column if not exists email text;

-- Strict case-insensitive uniqueness constraints (prevents duplicate users & emails)
create unique index if not exists idx_app_users_lower_username
    on public.app_users (lower(trim(username)));

create unique index if not exists idx_app_users_lower_email
    on public.app_users (lower(trim(email)))
    where email is not null and email <> '';

-- -----------------------------------------------------------------------------
-- 3. Automatic Password Hashing Trigger (Bcrypt via pgcrypto)
-- -----------------------------------------------------------------------------
create or replace function public.fn_auto_hash_password()
returns trigger as $$
begin
    if new.password is not null and (tg_op = 'INSERT' or new.password <> old.password) then
        -- Only hash if password is not already a valid bcrypt hash
        if not (new.password ~ '^\$2[aby]\$[0-9]{2}\$[./A-Za-z0-9]{53}$') then
            new.password := crypt(new.password, gen_salt('bf', 10));
        end if;
    end if;
    return new;
end;
$$ language plpgsql security definer;

drop trigger if exists trg_auto_hash_password on public.app_users;
create trigger trg_auto_hash_password
    before insert or update of password on public.app_users
    for each row execute procedure public.fn_auto_hash_password();

-- -----------------------------------------------------------------------------
-- 4. RPC: Authenticate User (Supports login via EITHER username OR email)
-- -----------------------------------------------------------------------------
drop function if exists public.authenticate_user(text, text);

create or replace function public.authenticate_user(
    p_username text,
    p_password text
) returns table (
    user_id uuid,
    username text,
    role text,
    email text
) language plpgsql security definer as $$
begin
    return query
    select u.id, u.username, u.role, u.email
    from public.app_users u
    where (
        lower(trim(u.username)) = lower(trim(p_username))
        or (u.email is not null and lower(trim(u.email)) = lower(trim(p_username)))
    )
    and u.password = crypt(p_password, u.password)
    limit 1;
end;
$$;

-- -----------------------------------------------------------------------------
-- 5. RPC: Check User Availability (Prevents duplicate usernames / emails)
-- -----------------------------------------------------------------------------
drop function if exists public.check_user_availability(text, text);

create or replace function public.check_user_availability(
    p_username text,
    p_email text
) returns json language plpgsql security definer as $$
declare
    clean_user text := lower(trim(coalesce(p_username, '')));
    clean_mail text := lower(trim(coalesce(p_email, '')));
begin
    -- Reserved system names
    if clean_user in ('developer', 'user', 'guest', 'admin', 'root', 'sanada', 'system', 'api') then
        return json_build_object(
            'available', false,
            'field', 'username',
            'error', 'The username "' || clean_user || '" is reserved. Please choose another.'
        );
    end if;

    -- Username collision check in public.app_users
    if clean_user <> '' then
        if exists (select 1 from public.app_users where lower(trim(username)) = clean_user) then
            return json_build_object(
                'available', false,
                'field', 'username',
                'error', 'Username is already taken. Please choose a different username.'
            );
        end if;
    end if;

    -- Email collision check in public.app_users
    if clean_mail <> '' then
        if exists (select 1 from public.app_users where lower(trim(coalesce(email, ''))) = clean_mail) then
            return json_build_object(
                'available', false,
                'field', 'email',
                'error', 'An account with this email already exists. Please sign in instead.'
            );
        end if;
    end if;

    -- Only check auth.users for confirmed accounts (unconfirmed signups should not block OTP resend)
    begin
        if clean_mail <> '' and exists (
            select 1 from auth.users
            where lower(trim(email)) = clean_mail
              and email_confirmed_at is not null
        ) then
            return json_build_object(
                'available', false,
                'field', 'email',
                'error', 'An account with this email already exists. Please sign in instead.'
            );
        end if;
    exception when others then
    end;

    return json_build_object('available', true);
end;
$$;

-- -----------------------------------------------------------------------------
-- 6. RPC: Register Verified User (Upsert on verified email, strict uniqueness on username)
-- -----------------------------------------------------------------------------
drop function if exists public.register_verified_user(text, text, text);

create or replace function public.register_verified_user(
    p_username text,
    p_password text,
    p_email text default null
) returns json language plpgsql security definer as $$
declare
    new_id uuid;
    clean_user text := trim(p_username);
    clean_mail text := lower(trim(coalesce(p_email, '')));
    existing_id uuid;
begin
    -- 1. Check reserved usernames
    if lower(clean_user) in ('developer', 'user', 'guest', 'admin', 'root', 'sanada', 'system', 'api') then
        return json_build_object('success', false, 'error', 'The username "' || clean_user || '" is reserved.');
    end if;

    -- 2. Check if username is taken by a DIFFERENT email
    if exists (
        select 1 from public.app_users
        where lower(trim(username)) = lower(clean_user)
          and (clean_mail = '' or email is null or lower(trim(email)) <> clean_mail)
    ) then
        return json_build_object('success', false, 'error', 'Username is already taken. Please choose a different username.');
    end if;

    -- 3. If account exists for this email (e.g. from prior creation), update with verified credentials
    if clean_mail <> '' then
        select id into existing_id
        from public.app_users
        where lower(trim(coalesce(email, ''))) = clean_mail
        limit 1;

        if existing_id is not null then
            update public.app_users
            set username = clean_user,
                password = p_password,
                role = 'user'
            where id = existing_id;

            return json_build_object('success', true, 'user_id', existing_id, 'username', clean_user, 'role', 'user');
        end if;
    end if;

    -- 4. Clean insert for new verified user
    insert into public.app_users (username, password, role, email)
    values (clean_user, p_password, 'user', nullif(clean_mail, ''))
    returning id into new_id;

    return json_build_object('success', true, 'user_id', new_id, 'username', clean_user, 'role', 'user');
exception
    when unique_violation then
        if clean_mail <> '' then
            update public.app_users
            set username = clean_user, password = p_password, role = 'user'
            where lower(trim(coalesce(email, ''))) = clean_mail;
            return json_build_object('success', true, 'username', clean_user, 'role', 'user');
        end if;
        return json_build_object('success', false, 'error', 'An account with this username or email already exists.');
    when others then
        return json_build_object('success', false, 'error', SQLERRM);
end;
$$;

-- -----------------------------------------------------------------------------
-- 7. RPC: Safe User Profile Lookup
-- -----------------------------------------------------------------------------
drop function if exists public.get_user_profile(text);

create or replace function public.get_user_profile(
    p_user_id text
) returns table (
    user_id uuid,
    username text,
    role text,
    email text,
    created_at timestamptz
) language plpgsql security definer as $$
declare
    v_uid uuid;
begin
    begin
        v_uid := p_user_id::uuid;
        return query
        select u.id, u.username, u.role, u.email, u.created_at
        from public.app_users u
        where u.id = v_uid;
        if found then return; end if;
    exception when others then
    end;

    return query
    select u.id, u.username, u.role, u.email, u.created_at
    from public.app_users u
    where lower(trim(u.username)) = lower(trim(p_user_id))
       or (u.email is not null and lower(trim(u.email)) = lower(trim(p_user_id)))
    limit 1;
end;
$$;

-- -----------------------------------------------------------------------------
-- 8. RPC: Change Password (Unified, fixes PostgREST ambiguous candidate error)
-- -----------------------------------------------------------------------------
-- Drop legacy UUID and text overloads before creating to avoid candidate ambiguity
drop function if exists public.change_password(uuid, text, text);
drop function if exists public.change_password(text, text, text);

create or replace function public.change_password(
    p_user_id text,
    p_old text,
    p_new text
) returns json language plpgsql security definer as $$
declare
    v_uid uuid;
    v_user public.app_users%rowtype;
    clean_old text := trim(coalesce(p_old, ''));
    clean_new text := trim(coalesce(p_new, ''));
begin
    -- 1. Try resolving user by UUID first, then fallback to username
    begin
        v_uid := p_user_id::uuid;
        select * into v_user from public.app_users where id = v_uid;
    exception when others then
        select * into v_user from public.app_users where lower(username) = lower(p_user_id);
    end;

    if v_user.id is null then
        select * into v_user from public.app_users where lower(username) = lower(p_user_id);
    end if;

    if v_user.id is null then
        return json_build_object('success', false, 'error', 'User account not found.');
    end if;

    -- 2. Verify current password with bcrypt
    if v_user.password is null or v_user.password <> crypt(clean_old, v_user.password) then
        return json_build_object('success', false, 'error', 'Current password is incorrect.');
    end if;

    -- 3. Password length validation
    if length(clean_new) < 8 then
        return json_build_object('success', false, 'error', 'New password must be at least 8 characters long.');
    end if;

    -- 4. Ensure new password differs from current
    if clean_old = clean_new then
        return json_build_object('success', false, 'error', 'New password must be different from current password.');
    end if;

    -- 5. Update password (trigger automatically handles bcrypt hashing)
    update public.app_users
    set password = clean_new
    where id = v_user.id;

    return json_build_object('success', true, 'message', 'Password updated successfully.');
exception when others then
    return json_build_object('success', false, 'error', SQLERRM);
end;
$$;

-- -----------------------------------------------------------------------------
-- 8b. RPC: Delete User Account (Requires Password Verification via Bcrypt)
-- -----------------------------------------------------------------------------
drop function if exists public.delete_user_account(text, text);

create or replace function public.delete_user_account(
    p_user_id text,
    p_password text
) returns json language plpgsql security definer as $$
declare
    v_uid uuid;
    v_user public.app_users%rowtype;
    clean_pass text := trim(coalesce(p_password, ''));
begin
    -- 1. Resolve user by UUID first, then username or email
    begin
        v_uid := p_user_id::uuid;
        select * into v_user from public.app_users where id = v_uid;
    exception when others then
        select * into v_user from public.app_users
        where lower(trim(username)) = lower(trim(p_user_id))
           or (email is not null and lower(trim(email)) = lower(trim(p_user_id)))
        limit 1;
    end;

    if v_user.id is null then
        select * into v_user from public.app_users
        where lower(trim(username)) = lower(trim(p_user_id))
           or (email is not null and lower(trim(email)) = lower(trim(p_user_id)))
        limit 1;
    end if;

    if v_user.id is null then
        return json_build_object('success', false, 'error', 'User account not found.');
    end if;

    -- 2. Protect built-in demo accounts
    if lower(v_user.username) in ('developer', 'user', 'admin') and (v_user.email is null or v_user.email = '') then
        return json_build_object('success', false, 'error', 'Built-in demo accounts cannot be deleted.');
    end if;

    -- 3. Verify password via pgcrypto crypt()
    if v_user.password is null or v_user.password <> crypt(clean_pass, v_user.password) then
        return json_build_object('success', false, 'error', 'Incorrect password. Account deletion aborted.');
    end if;

    -- 4. Delete associated records
    delete from public.api_keys where user_id = v_user.id or lower(trim(user_identifier)) = lower(trim(v_user.username));
    delete from public.run_history where user_id = v_user.id or lower(trim(user_identifier)) = lower(trim(v_user.username));
    delete from public.app_users where id = v_user.id;

    -- 5. Cascade delete from auth.users if email exists
    begin
        if v_user.email is not null and v_user.email <> '' then
            delete from auth.users where lower(trim(email)) = lower(trim(v_user.email));
        end if;
    exception when others then
    end;

    return json_build_object('success', true, 'message', 'Account deleted successfully.');
exception when others then
    return json_build_object('success', false, 'error', SQLERRM);
end;
$$;

-- -----------------------------------------------------------------------------
-- 9. Table: public.api_keys & Procedure: verify_api_key
-- -----------------------------------------------------------------------------
create table if not exists public.api_keys (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references public.app_users(id) on delete cascade,
    user_identifier text,
    name text not null default 'My app key 1',
    key_prefix text not null,
    key_hash text not null,
    masked_key text not null,
    encrypted_key text,
    created_at timestamptz not null default now()
);

alter table public.api_keys add column if not exists user_identifier text;

create index if not exists idx_api_keys_hash on public.api_keys(key_hash);
create index if not exists idx_api_keys_user_id on public.api_keys(user_id);
create index if not exists idx_api_keys_user_identifier on public.api_keys(user_identifier);

alter table public.api_keys enable row level security;

drop policy if exists "Allow select api_keys" on public.api_keys;
create policy "Allow select api_keys" on public.api_keys for select using (true);

drop policy if exists "Allow insert api_keys" on public.api_keys;
create policy "Allow insert api_keys" on public.api_keys for insert with check (true);

drop policy if exists "Allow delete api_keys" on public.api_keys;
create policy "Allow delete api_keys" on public.api_keys for delete using (true);

drop function if exists public.verify_api_key(text);

create or replace function public.verify_api_key(p_key_hash text)
returns table (
    is_valid boolean,
    key_name text,
    user_id uuid,
    user_identifier text,
    created_at timestamptz
) language plpgsql security definer as $$
begin
    return query
    select
        true as is_valid,
        k.name as key_name,
        k.user_id,
        k.user_identifier,
        k.created_at
    from public.api_keys k
    where k.key_hash = p_key_hash
    limit 1;
end;
$$;

-- -----------------------------------------------------------------------------
-- 10. Table: public.run_history & Procedures (Max 5 runs per user)
-- -----------------------------------------------------------------------------
create table if not exists public.run_history (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references public.app_users(id) on delete cascade,
    user_identifier text not null,
    name text not null,
    filename text not null,
    mode text not null default 'auto',
    threshold numeric not null default 0.5,
    task text not null default 'all',
    all_4_stems boolean not null default false,
    is_pinned boolean not null default false,
    result_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_run_history_user_id on public.run_history(user_id);
create index if not exists idx_run_history_user_ident on public.run_history(user_identifier);
create index if not exists idx_run_history_created_at on public.run_history(created_at desc);

alter table public.run_history enable row level security;

drop policy if exists "Allow select run_history" on public.run_history;
create policy "Allow select run_history" on public.run_history for select using (true);

drop policy if exists "Allow insert run_history" on public.run_history;
create policy "Allow insert run_history" on public.run_history for insert with check (true);

drop policy if exists "Allow update run_history" on public.run_history;
create policy "Allow update run_history" on public.run_history for update using (true);

drop policy if exists "Allow delete run_history" on public.run_history;
create policy "Allow delete run_history" on public.run_history for delete using (true);

drop function if exists public.save_run_history(uuid, text, text, text, text, numeric, text, boolean, jsonb);

create or replace function public.save_run_history(
    p_user_id uuid,
    p_user_identifier text,
    p_name text,
    p_filename text,
    p_mode text,
    p_threshold numeric,
    p_task text,
    p_all_4_stems boolean,
    p_result_json jsonb
) returns json language plpgsql security definer as $$
declare
    new_id uuid;
    clean_name text;
    clean_ident text;
begin
    clean_ident := lower(trim(coalesce(p_user_identifier, 'user')));
    clean_name := trim(coalesce(p_name, p_filename, 'Neural Inference Run'));

    -- 1. Insert new run
    insert into public.run_history (
        user_id,
        user_identifier,
        name,
        filename,
        mode,
        threshold,
        task,
        all_4_stems,
        result_json,
        created_at
    ) values (
        p_user_id,
        clean_ident,
        clean_name,
        p_filename,
        coalesce(p_mode, 'auto'),
        coalesce(p_threshold, 0.5),
        coalesce(p_task, 'all'),
        coalesce(p_all_4_stems, false),
        p_result_json,
        now()
    ) returning id into new_id;

    -- 2. Strictly keep latest 5 runs per user
    delete from public.run_history
    where (
        (p_user_id is not null and user_id = p_user_id)
        or (user_identifier = clean_ident)
    )
    and id not in (
        select id from public.run_history
        where (
            (p_user_id is not null and user_id = p_user_id)
            or (user_identifier = clean_ident)
        )
        order by created_at desc
        limit 5
    );

    return json_build_object(
        'success', true,
        'run_id', new_id,
        'name', clean_name,
        'user_identifier', clean_ident
    );
exception when others then
    return json_build_object('success', false, 'error', SQLERRM);
end;
$$;

drop function if exists public.get_run_history(uuid, text);

create or replace function public.get_run_history(
    p_user_id uuid,
    p_user_identifier text
) returns table (
    id uuid,
    user_id uuid,
    user_identifier text,
    name text,
    filename text,
    mode text,
    threshold numeric,
    task text,
    all_4_stems boolean,
    is_pinned boolean,
    result_json jsonb,
    created_at timestamptz
) language plpgsql security definer as $$
declare
    clean_ident text;
begin
    clean_ident := lower(trim(coalesce(p_user_identifier, '')));
    return query
    select
        r.id,
        r.user_id,
        r.user_identifier,
        r.name,
        r.filename,
        r.mode,
        r.threshold,
        r.task,
        r.all_4_stems,
        r.is_pinned,
        r.result_json,
        r.created_at
    from public.run_history r
    where (
        (p_user_id is not null and r.user_id = p_user_id)
        or (clean_ident <> '' and r.user_identifier = clean_ident)
    )
    order by r.created_at desc
    limit 5;
end;
$$;

drop function if exists public.rename_run_history(uuid, text, uuid, text);

create or replace function public.rename_run_history(
    p_run_id uuid,
    p_name text,
    p_user_id uuid,
    p_user_identifier text
) returns json language plpgsql security definer as $$
declare
    clean_ident text;
    clean_name text;
begin
    clean_ident := lower(trim(coalesce(p_user_identifier, '')));
    clean_name := trim(coalesce(p_name, 'Inference Run'));

    update public.run_history
    set name = clean_name
    where id = p_run_id
    and (
        (p_user_id is not null and user_id = p_user_id)
        or (clean_ident <> '' and user_identifier = clean_ident)
    );

    return json_build_object('success', true, 'name', clean_name);
end;
$$;

drop function if exists public.delete_run_history(uuid, uuid, text);

create or replace function public.delete_run_history(
    p_run_id uuid,
    p_user_id uuid,
    p_user_identifier text
) returns json language plpgsql security definer as $$
declare
    clean_ident text;
begin
    clean_ident := lower(trim(coalesce(p_user_identifier, '')));

    delete from public.run_history
    where id = p_run_id
    and (
        (p_user_id is not null and user_id = p_user_id)
        or (clean_ident <> '' and user_identifier = clean_ident)
    );

    return json_build_object('success', true);
end;
$$;

drop function if exists public.clear_run_history(uuid, text);

create or replace function public.clear_run_history(
    p_user_id uuid,
    p_user_identifier text
) returns json language plpgsql security definer as $$
declare
    clean_ident text;
begin
    clean_ident := lower(trim(coalesce(p_user_identifier, '')));

    delete from public.run_history
    where (
        (p_user_id is not null and user_id = p_user_id)
        or (clean_ident <> '' and user_identifier = clean_ident)
    );

    return json_build_object('success', true);
end;
$$;

-- -----------------------------------------------------------------------------
-- 11. Cleanup Deprecated Tables/Functions
-- -----------------------------------------------------------------------------
drop function if exists public.toggle_pin_run_history(uuid, uuid, text);
drop table if exists public.user_preferences cascade;
drop function if exists public.save_user_preferences cascade;
drop function if exists public.get_user_preferences cascade;

-- -----------------------------------------------------------------------------
-- 12. Permissions (Grants to anon, authenticated, and service_role)
-- -----------------------------------------------------------------------------
grant all on table public.app_users to anon, authenticated, service_role;
grant all on table public.api_keys to anon, authenticated, service_role;
grant all on table public.run_history to anon, authenticated, service_role;

grant execute on function public.authenticate_user(text, text) to anon, authenticated, service_role;
grant execute on function public.check_user_availability(text, text) to anon, authenticated, service_role;
grant execute on function public.register_verified_user(text, text, text) to anon, authenticated, service_role;
grant execute on function public.get_user_profile(text) to anon, authenticated, service_role;
grant execute on function public.change_password(text, text, text) to anon, authenticated, service_role;
grant execute on function public.delete_user_account(text, text) to anon, authenticated, service_role;
grant execute on function public.verify_api_key(text) to anon, authenticated, service_role;
grant execute on function public.save_run_history(uuid, text, text, text, text, numeric, text, boolean, jsonb) to anon, authenticated, service_role;
grant execute on function public.get_run_history(uuid, text) to anon, authenticated, service_role;
grant execute on function public.rename_run_history(uuid, text, uuid, text) to anon, authenticated, service_role;
grant execute on function public.delete_run_history(uuid, uuid, text) to anon, authenticated, service_role;
grant execute on function public.clear_run_history(uuid, text) to anon, authenticated, service_role;
