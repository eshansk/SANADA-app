-- =============================================================================
-- SANADA Complete Supabase Setup & Trigger Fix
-- =============================================================================
-- Paste and run this complete query in your Supabase SQL Editor.
-- =============================================================================

-- 1. Drop old triggers on auth.users (fixes "Database error saving new user")
drop trigger if exists on_auth_user_created on auth.users;
drop function if exists public.handle_new_user() cascade;
drop function if exists public.handle_new_auth_user() cascade;

-- 2. Add email column to public.app_users (safe if column already exists)
alter table public.app_users
add column if not exists email text unique;

create unique index if not exists idx_app_users_lower_username
on public.app_users (lower(trim(username)));

create unique index if not exists idx_app_users_lower_email
on public.app_users (lower(trim(email)))
where email is not null and email <> '';

-- Helper RPC: Check username & email availability before sending OTP
create or replace function public.check_user_availability(
    p_username text,
    p_email text
) returns json language plpgsql security definer as $$
declare
    clean_user text := lower(trim(coalesce(p_username, '')));
    clean_mail text := lower(trim(coalesce(p_email, '')));
begin
    -- 1. Check reserved system usernames
    if clean_user in ('developer', 'user', 'guest', 'admin', 'root', 'sanada', 'system', 'api') then
        return json_build_object(
            'available', false,
            'field', 'username',
            'error', 'The username "' || clean_user || '" is reserved. Please choose another.'
        );
    end if;

    -- 2. Check username collision (case-insensitive) in public.app_users
    if clean_user <> '' then
        if exists (select 1 from public.app_users where lower(trim(username)) = clean_user) then
            return json_build_object(
                'available', false,
                'field', 'username',
                'error', 'Username is already taken. Please choose a different username.'
            );
        end if;
    end if;

    -- 3. Check email collision (case-insensitive) in public.app_users
    if clean_mail <> '' then
        if exists (select 1 from public.app_users where lower(trim(coalesce(email, ''))) = clean_mail) then
            return json_build_object(
                'available', false,
                'field', 'email',
                'error', 'An account with this email already exists. Please sign in instead.'
            );
        end if;
    end if;

    -- 4. Check auth.users if accessible
    begin
        if clean_mail <> '' and exists (select 1 from auth.users where lower(trim(email)) = clean_mail) then
            return json_build_object(
                'available', false,
                'field', 'email',
                'error', 'An account with this email already exists. Please sign in instead.'
            );
        end if;
        if clean_user <> '' and exists (select 1 from auth.users where lower(trim(raw_user_meta_data->>'username')) = clean_user) then
            return json_build_object(
                'available', false,
                'field', 'username',
                'error', 'Username is already taken. Please choose a different username.'
            );
        end if;
    exception when others then
        -- Ignore schema restriction if auth.users not accessible
    end;

    return json_build_object('available', true);
end;
$$;

-- 3. Procedure to register verified user into public.app_users
create or replace function public.register_verified_user (
  p_username text,
  p_password text,
  p_email text default null
) returns json language plpgsql security definer as $$
declare
  new_id uuid;
  clean_user text := trim(p_username);
  clean_mail text := lower(trim(coalesce(p_email, '')));
begin
  if lower(clean_user) in ('developer', 'user', 'guest', 'admin', 'root', 'sanada', 'system', 'api') then
    return json_build_object('success', false, 'error', 'The username "' || clean_user || '" is reserved.');
  end if;

  if exists (select 1 from public.app_users where lower(trim(username)) = lower(clean_user)) then
    return json_build_object('success', false, 'error', 'Username is already taken. Please choose a different username.');
  end if;

  if clean_mail <> '' and exists (select 1 from public.app_users where lower(trim(coalesce(email, ''))) = clean_mail) then
    return json_build_object('success', false, 'error', 'An account with this email already exists. Please sign in instead.');
  end if;

  insert into public.app_users (username, password, role, email)
  values (clean_user, p_password, 'user', nullif(clean_mail, ''))
  returning id into new_id;

  return json_build_object('success', true, 'user_id', new_id, 'username', clean_user, 'role', 'user'); 
exception
  when unique_violation then 
    return json_build_object('success', false, 'error', 'An account with this username or email already exists.');
  when others then
    return json_build_object('success', false, 'error', SQLERRM);
end;
$$;

-- 4. Allow login with EITHER username OR email
create or replace function public.authenticate_user (
  p_username text,
  p_password text
) returns table (user_id uuid, username text, role text) language plpgsql security definer as $$ 
begin
  return query
  select u.id, u.username, u.role
  from public.app_users u
  where (
    lower(trim(u.username)) = lower(trim(p_username))
    or (u.email is not null and lower(trim(u.email)) = lower(trim(p_username)))
  )
  and u.password = crypt(p_password, u.password);
end;
$$;

-- 5. Drop any obsolete auth triggers on auth.users (fixes "Database error saving new user")
drop trigger if exists on_auth_user_created on auth.users;
drop function if exists public.handle_new_user() cascade;
drop function if exists public.handle_new_auth_user() cascade;

-- 6. Grant permissions
grant execute on function public.check_user_availability (text, text) to anon, authenticated, service_role;
grant execute on function public.register_verified_user (text, text, text) to anon, authenticated, service_role;
grant execute on function public.authenticate_user (text, text) to anon, authenticated, service_role;
