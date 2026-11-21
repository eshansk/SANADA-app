-- =============================================================================
-- SANADA Run History & Password Management Schema
-- =============================================================================
-- Run this script in your Supabase SQL Editor.
-- Provides:
--  1. public.run_history table with user-scoping and RLS
--  2. Auto-maintenance keeping strictly the latest 5 runs per user (chronological)
--  3. RPCs: save_run_history, get_run_history, delete_run_history, clear_run_history,
--     rename_run_history
--  4. RPC: change_password with crypt() verification and double-hash prevention
-- =============================================================================

create extension if not exists pgcrypto;

-- -----------------------------------------------------------------------------
-- 1. Table: public.run_history
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

-- Indexes for high-performance per-user queries
create index if not exists idx_run_history_user_id on public.run_history(user_id);
create index if not exists idx_run_history_user_ident on public.run_history(user_identifier);
create index if not exists idx_run_history_created_at on public.run_history(created_at desc);
drop index if exists public.idx_run_history_pinned;

-- -----------------------------------------------------------------------------
-- 2. Row Level Security (RLS)
-- -----------------------------------------------------------------------------
alter table public.run_history enable row level security;

-- Policies for public.run_history
drop policy if exists "Allow select run_history" on public.run_history;
create policy "Allow select run_history" on public.run_history
    for select using (true);

drop policy if exists "Allow insert run_history" on public.run_history;
create policy "Allow insert run_history" on public.run_history
    for insert with check (true);

drop policy if exists "Allow update run_history" on public.run_history;
create policy "Allow update run_history" on public.run_history
    for update using (true);

drop policy if exists "Allow delete run_history" on public.run_history;
create policy "Allow delete run_history" on public.run_history
    for delete using (true);

grant all on table public.run_history to anon, authenticated, service_role;

-- -----------------------------------------------------------------------------
-- 3. Procedure: Save Run History (Maintains strictly max 5 runs per user)
-- -----------------------------------------------------------------------------
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

    -- 1. Insert new run record
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

    -- 2. Enforce invariant: keep ONLY latest 5 runs per user (strictly newest first)
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

-- -----------------------------------------------------------------------------
-- 4. Procedure: Get Run History (Latest 5 runs)
-- -----------------------------------------------------------------------------
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

-- -----------------------------------------------------------------------------
-- 5. Procedure: Rename Run History
-- -----------------------------------------------------------------------------
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

-- -----------------------------------------------------------------------------
-- 6. Pin Feature Deprecation & Removal
-- -----------------------------------------------------------------------------
drop function if exists public.toggle_pin_run_history(uuid, uuid, text);

-- -----------------------------------------------------------------------------
-- 7. Procedure: Delete Single Run
-- -----------------------------------------------------------------------------
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

-- -----------------------------------------------------------------------------
-- 8. Procedure: Clear All Runs
-- -----------------------------------------------------------------------------
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
-- 9. Password Auto-Hashing Trigger (Guards against double-hashing)
-- -----------------------------------------------------------------------------
create or replace function public.fn_auto_hash_password()
returns trigger as $$
begin
    if new.password is not null and (tg_op = 'INSERT' or new.password <> old.password) then
        -- Only crypt() if the password is not already a valid bcrypt hash
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
-- 10. Procedure: Change Password
-- -----------------------------------------------------------------------------
-- Overload 1: UUID user id
create or replace function public.change_password(
    p_user_id uuid,
    p_old text,
    p_new text
) returns json language plpgsql security definer as $$
declare
    v_user public.app_users%rowtype;
    clean_old text := trim(coalesce(p_old, ''));
    clean_new text := trim(coalesce(p_new, ''));
begin
    -- 1. Find user by id
    select * into v_user from public.app_users where id = p_user_id;
    if v_user.id is null then
        return json_build_object('success', false, 'error', 'User account not found.');
    end if;

    -- 2. Verify current password
    if v_user.password is null or v_user.password <> crypt(clean_old, v_user.password) then
        return json_build_object('success', false, 'error', 'Current password is incorrect.');
    end if;

    -- 3. Validate new password length (minimum 8 characters)
    if length(clean_new) < 8 then
        return json_build_object('success', false, 'error', 'New password must be at least 8 characters long.');
    end if;

    -- 4. Validate that new password differs from current
    if clean_old = clean_new then
        return json_build_object('success', false, 'error', 'New password must be different from current password.');
    end if;

    -- 5. Update password
    -- Because trg_auto_hash_password checks for existing bcrypt hashes, setting plain
    -- clean_new triggers exactly one bcrypt hash without double-hashing.
    update public.app_users
    set password = clean_new
    where id = p_user_id;

    return json_build_object('success', true, 'message', 'Password updated successfully.');
exception when others then
    return json_build_object('success', false, 'error', SQLERRM);
end;
$$;

-- Overload 2: Text user id / username for flexible client callers
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
    -- Try resolving as UUID first
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

    if v_user.password is null or v_user.password <> crypt(clean_old, v_user.password) then
        return json_build_object('success', false, 'error', 'Current password is incorrect.');
    end if;

    if length(clean_new) < 8 then
        return json_build_object('success', false, 'error', 'New password must be at least 8 characters long.');
    end if;

    if clean_old = clean_new then
        return json_build_object('success', false, 'error', 'New password must be different from current password.');
    end if;

    update public.app_users
    set password = clean_new
    where id = v_user.id;

    return json_build_object('success', true, 'message', 'Password updated successfully.');
exception when others then
    return json_build_object('success', false, 'error', SQLERRM);
end;
$$;

-- -----------------------------------------------------------------------------
-- 11. Grant Permissions
-- -----------------------------------------------------------------------------
grant execute on function public.save_run_history(uuid, text, text, text, text, numeric, text, boolean, jsonb) to anon, authenticated, service_role;
grant execute on function public.get_run_history(uuid, text) to anon, authenticated, service_role;
grant execute on function public.rename_run_history(uuid, text, uuid, text) to anon, authenticated, service_role;
grant execute on function public.delete_run_history(uuid, uuid, text) to anon, authenticated, service_role;
grant execute on function public.clear_run_history(uuid, text) to anon, authenticated, service_role;
grant execute on function public.change_password(uuid, text, text) to anon, authenticated, service_role;
grant execute on function public.change_password(text, text, text) to anon, authenticated, service_role;

-- -----------------------------------------------------------------------------
-- 12. Cleanup (User preferences table/functions are deprecated & removed)
-- -----------------------------------------------------------------------------
drop table if exists public.user_preferences cascade;
drop function if exists public.save_user_preferences cascade;
drop function if exists public.get_user_preferences cascade;

-- -----------------------------------------------------------------------------
-- 13. Optional Supabase Storage Bucket for Stems
-- -----------------------------------------------------------------------------
-- If you wish to use Supabase Storage in addition to embedded JSONB audio:
-- insert into storage.buckets (id, name, public)
-- values ('sanada-stems', 'sanada-stems', true)
-- on conflict (id) do nothing;
--
-- create policy "Allow public stem downloads"
-- on storage.objects for select
-- using (bucket_id = 'sanada-stems');
