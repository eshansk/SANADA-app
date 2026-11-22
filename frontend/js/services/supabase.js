/**
 * frontend/js/services/supabase.js
 * ================================
 * Supabase Client & Authentication Service for SANADA.
 *
 * Integrates directly with your Supabase PostgreSQL database:
 *  - Table: public.app_users (username, password, role)
 *  - Trigger: trg_auto_hash_password (automatic bcrypt password encryption)
 *  - RPC Procedure: public.authenticate_user(p_username, p_password)
 *
 * CONFIGURATION INSTRUCTIONS:
 * ---------------------------
 * In your Supabase Dashboard:
 * 1. Navigate to Project Settings -> API.
 * 2. Copy the "Project URL" and paste it into SUPABASE_URL below.
 * 3. Copy the "Project API keys" -> anon / public key and paste it into SUPABASE_ANON_KEY below.
 *
 * You can also set them dynamically in the browser console via:
 *   window.setSupabaseConfig('https://your-project.supabase.co', 'your-anon-key');
 */

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm';

// =============================================================================
// 1. CREDENTIALS CONFIGURATION
// =============================================================================
export const SUPABASE_URL = (typeof window !== 'undefined' && window.SANADA_SUPABASE_URL)
    || (typeof localStorage !== 'undefined' && localStorage.getItem('sanada_supabase_url'))
    || 'https://elgsaepnceawrkswdpnv.supabase.co';

export const SUPABASE_ANON_KEY = (typeof window !== 'undefined' && window.SANADA_SUPABASE_ANON_KEY)
    || (typeof localStorage !== 'undefined' && localStorage.getItem('sanada_supabase_anon_key'))
    || 'sb_publishable_O-6zrNb2byz6he6Lx2Wg2g_pZKTicN1';

/**
 * Returns the currently active Supabase URL (from code, window, or localStorage).
 */
export function getSupabaseUrl() {
    if (typeof window !== 'undefined' && window.SANADA_SUPABASE_URL) return window.SANADA_SUPABASE_URL;
    if (typeof localStorage !== 'undefined') {
        const stored = localStorage.getItem('sanada_supabase_url');
        if (stored) return stored.trim();
    }
    return SUPABASE_URL;
}

/**
 * Returns the currently active Supabase Anon Key (from code, window, or localStorage).
 */
export function getSupabaseKey() {
    if (typeof window !== 'undefined' && window.SANADA_SUPABASE_ANON_KEY) return window.SANADA_SUPABASE_ANON_KEY;
    if (typeof localStorage !== 'undefined') {
        const stored = localStorage.getItem('sanada_supabase_anon_key');
        if (stored) return stored.trim();
    }
    return SUPABASE_ANON_KEY;
}

/**
 * Checks if Supabase credentials have been configured with valid non-placeholder values.
 * @returns {boolean}
 */
export function isSupabaseConfigured() {
    const url = getSupabaseUrl();
    const key = getSupabaseKey();
    return Boolean(
        url
        && typeof url === 'string'
        && url.startsWith('https://')
        && !url.includes('YOUR_PROJECT_ID')
        && !url.includes('YOUR_SUPABASE')
        && key
        && typeof key === 'string'
        && !key.includes('YOUR_SUPABASE_ANON_KEY')
        && key.length > 20
    );
}

// =============================================================================
// 2. SUPABASE CLIENT SINGLETON
// =============================================================================
let _clientInstance = null;

/**
 * Lazily instantiates and returns the Supabase client.
 * Returns null if credentials are not configured.
 */
export function getSupabaseClient() {
    if (_clientInstance) return _clientInstance;

    if (!isSupabaseConfigured()) {
        return null;
    }

    try {
        const url = getSupabaseUrl();
        const key = getSupabaseKey();
        _clientInstance = createClient(url, key, {
            auth: {
                persistSession: false,
                autoRefreshToken: false,
                detectSessionInUrl: false
            }
        });
        return _clientInstance;
    } catch (err) {
        console.error('[SANADA Supabase] Client initialization failed:', err);
        return null;
    }
}

export const supabase = getSupabaseClient();

// =============================================================================
// 3. USER AUTHENTICATION & REGISTRATION RPC / AUTH
// =============================================================================
/**
 * Authenticates a user against public.authenticate_user(p_username, p_password),
 * with fallback to client.auth.signInWithPassword for Supabase Auth accounts.
 *
 * @param {string} usernameOrEmail - User account username or email
 * @param {string} password - Raw plain-text password
 * @returns {Promise<{
 *   success: boolean,
 *   user?: { id: string, username: string, role: string, email?: string },
 *   error?: string
 * }>}
 */
export async function loginWithSupabase(usernameOrEmail, password) {
    const client = getSupabaseClient();
    if (!client) {
        return {
            success: false,
            error: 'Supabase credentials are not configured yet. Paste your SUPABASE_URL and SUPABASE_ANON_KEY into frontend/js/services/supabase.js'
        };
    }

    const cleanUser = (usernameOrEmail || '').trim();
    const cleanPass = (password || '').trim();

    if (!cleanUser || !cleanPass) {
        return {
            success: false,
            error: 'Please enter both username and password.'
        };
    }

    try {
        // 1. First attempt: PostgreSQL authenticate_user RPC function
        const { data, error } = await client.rpc('authenticate_user', {
            p_username: cleanUser,
            p_password: cleanPass
        });

        if (!error && Array.isArray(data) && data.length > 0) {
            const match = data[0];
            let userEmail = match.email || null;
            let createdAt = null;

            // Fetch actual email from get_user_profile RPC or app_users table
            try {
                const { data: prof } = await client.rpc('get_user_profile', {
                    p_user_id: String(match.user_id || match.id || cleanUser)
                });
                if (Array.isArray(prof) && prof.length > 0 && prof[0].email) {
                    userEmail = prof[0].email;
                    if (prof[0].created_at) createdAt = prof[0].created_at;
                }
            } catch (pErr) {}

            if (!userEmail) {
                try {
                    const { data: userData } = await client
                        .from('app_users')
                        .select('email, created_at')
                        .eq('id', match.user_id || match.id)
                        .maybeSingle();
                    if (userData) {
                        if (userData.email) userEmail = userData.email;
                        if (userData.created_at) createdAt = userData.created_at;
                    }
                } catch (uErr) {}
            }

            if (!userEmail && cleanUser.includes('@')) {
                userEmail = cleanUser;
            }

            return {
                success: true,
                user: {
                    id: match.user_id || match.id || null,
                    username: match.username,
                    email: userEmail,
                    created_at: createdAt,
                    role: (match.role || 'user').toLowerCase()
                }
            };
        }

        // 2. Second attempt: Check Supabase Auth (for accounts created with email & password)
        try {
            const authRes = await client.auth.signInWithPassword({
                email: cleanUser,
                password: cleanPass
            });

            if (!authRes.error && authRes.data && authRes.data.user) {
                const suUser = authRes.data.user;
                const metadata = suUser.user_metadata || {};
                const resolvedUsername = metadata.username || (suUser.email ? suUser.email.split('@')[0] : 'user');
                const resolvedRole = (metadata.role || 'user').toLowerCase();
                return {
                    success: true,
                    user: {
                        id: suUser.id,
                        username: resolvedUsername,
                        email: suUser.email,
                        role: resolvedRole
                    }
                };
            }
        } catch (authErr) {
            // Ignore auth fallback error, proceed to invalid credentials response
        }

        // If authenticate_user had an actual code error (e.g. function missing), report it
        if (error && error.code === '42883') {
            return {
                success: false,
                error: 'Database function authenticate_user() not found. Please run the SQL setup script in Supabase SQL Editor.'
            };
        }

        return {
            success: false,
            error: 'Invalid username or password.'
        };
    } catch (err) {
        console.error('[SANADA Supabase] Unexpected network error:', err);
        return {
            success: false,
            error: err.message || 'Network error while contacting Supabase. Please check your internet connection.'
        };
    }
}

/**
 * Initiates user registration by creating an account in Supabase Auth
 * and triggering an email with a 6-digit OTP confirmation token.
 *
 * @param {string} username - Chosen username
 * @param {string} email - Valid email address to receive OTP
 * @param {string} password - Plain-text password (min 6 chars)
 * @returns {Promise<{ success: boolean, user?: any, error?: string }>}
 */
export async function signUpWithSupabase(username, email, password) {
    const client = getSupabaseClient();
    if (!client) {
        return {
            success: false,
            error: 'Supabase credentials are not configured yet.'
        };
    }

    const cleanUser = (username || '').trim();
    const cleanEmail = (email || '').trim().toLowerCase();
    const cleanPass = (password || '').trim();

    if (!cleanUser) {
        return { success: false, error: 'Please enter a username.' };
    }
    if (cleanUser.length < 3) {
        return { success: false, error: 'Username must be at least 3 characters long.' };
    }
    if (!/^[a-zA-Z0-9_-]+$/.test(cleanUser)) {
        return { success: false, error: 'Username can only contain letters, numbers, underscores, and hyphens.' };
    }

    const RESERVED_USERNAMES = ['developer', 'user', 'guest', 'admin', 'root', 'sanada', 'system', 'api'];
    if (RESERVED_USERNAMES.includes(cleanUser.toLowerCase())) {
        return {
            success: false,
            error: `The username "${cleanUser}" is reserved. Please choose a different username.`
        };
    }

    if (!cleanEmail || !cleanEmail.includes('@') || !cleanEmail.includes('.')) {
        return { success: false, error: 'Please enter a valid email address.' };
    }

    if (!cleanPass) {
        return { success: false, error: 'Please enter a password.' };
    }
    if (cleanPass.length < 6) {
        return { success: false, error: 'Password must be at least 6 characters long.' };
    }

    // 1. Availability check via RPC check_user_availability (enforcing case-insensitive uniqueness)
    try {
        const { data: avail, error: availErr } = await client.rpc('check_user_availability', {
            p_username: cleanUser,
            p_email: cleanEmail
        });
        if (!availErr && avail && avail.available === false) {
            return {
                success: false,
                error: avail.error || (avail.field === 'email'
                    ? 'An account with this email already exists. Please sign in instead.'
                    : 'Username is already taken. Please choose a different username.')
            };
        }
    } catch (e) {
        // Proceed to direct table check fallback if RPC not installed yet
    }

    // 2. Direct table fallback check on public.app_users (case-insensitive)
    try {
        const { data: uMatch } = await client
            .from('app_users')
            .select('id, username')
            .ilike('username', cleanUser)
            .maybeSingle();

        if (uMatch) {
            return {
                success: false,
                error: 'Username is already taken. Please choose a different username.'
            };
        }

        const { data: eMatch } = await client
            .from('app_users')
            .select('id, email')
            .ilike('email', cleanEmail)
            .maybeSingle();

        if (eMatch) {
            return {
                success: false,
                error: 'An account with this email already exists. Please sign in instead.'
            };
        }
    } catch (tblErr) {
        console.warn('[SANADA Supabase] Availability check notice:', tblErr);
    }

    try {
        const { data, error } = await client.auth.signUp({
            email: cleanEmail,
            password: cleanPass,
            options: {
                data: {
                    username: cleanUser,
                    role: 'user'
                }
            }
        });

        if (error) {
            console.error('[SANADA Supabase] Sign up error:', error);
            let friendlyMsg = error.message;
            const errStr = (error.message || '').toLowerCase();
            if (error.status === 429 || errStr.includes('rate limit')) {
                friendlyMsg = 'Email rate limit reached. Please wait a short moment before requesting another code.';
            } else if (errStr.includes('already registered')) {
                // Check if user is in public.app_users:
                // If NOT in public.app_users, they never finished OTP verification!
                // Resend OTP code so they can verify and complete registration!
                try {
                    const resendRes = await resendOtpWithSupabase(cleanEmail);
                    if (resendRes.success) {
                        return {
                            success: true,
                            email: cleanEmail,
                            username: cleanUser,
                            resend: true
                        };
                    }
                } catch (rErr) {}
                friendlyMsg = 'An account with this email already exists. Please sign in instead.';
            } else if (errStr.includes('database error saving new user')) {
                friendlyMsg = 'Database trigger error on auth.users. Please run the SQL fix in Supabase to drop on_auth_user_created.';
            } else if (errStr.includes('invalid') && errStr.includes('email')) {
                friendlyMsg = 'Please enter a valid email address (disposable email domains are blocked by Supabase).';
            }
            return {
                success: false,
                error: friendlyMsg
            };
        }

        // If identities is an empty array, the user exists in auth.users
        if (data.user && Array.isArray(data.user.identities) && data.user.identities.length === 0) {
            // Check if user already completed verification in public.app_users
            let userInApp = false;
            try {
                const { data: appUser } = await client
                    .from('app_users')
                    .select('id')
                    .ilike('email', cleanEmail)
                    .maybeSingle();
                if (appUser) userInApp = true;
            } catch (uErr) {}

            if (userInApp) {
                return {
                    success: false,
                    error: 'An account with this email already exists. Please sign in instead.'
                };
            }

            // User exists in auth.users as unconfirmed, but not in public.app_users.
            // Resend OTP code so they can complete registration!
            const resendRes = await resendOtpWithSupabase(cleanEmail);
            if (resendRes.success) {
                return {
                    success: true,
                    user: data.user,
                    email: cleanEmail,
                    username: cleanUser,
                    resend: true
                };
            }

            return {
                success: false,
                error: 'An account with this email already exists. Please sign in instead.'
            };
        }

        return {
            success: true,
            user: data.user,
            email: cleanEmail,
            username: cleanUser
        };
    } catch (err) {
        console.error('[SANADA Supabase] Unexpected sign up error:', err);
        return {
            success: false,
            error: err.message || 'Network error during account registration.'
        };
    }
}

/**
 * Verifies the 6-digit OTP code sent to the user's email during registration.
 *
 * @param {string} email - User email address
 * @param {string} token - 6-digit OTP code entered by the user
 * @param {string} username - Username associated with registration
 * @param {string} password - Password used during registration (to sync to app_users)
 * @returns {Promise<{
 *   success: boolean,
 *   user?: { id: string, username: string, email: string, role: string },
 *   session?: any,
 *   error?: string
 * }>}
 */
export async function verifyOtpWithSupabase(email, token, username, password) {
    const client = getSupabaseClient();
    if (!client) {
        return {
            success: false,
            error: 'Supabase credentials are not configured.'
        };
    }

    const cleanEmail = (email || '').trim().toLowerCase();
    const cleanToken = (token || '').trim();

    if (!cleanEmail) {
        return { success: false, error: 'Missing email address for verification.' };
    }
    if (!cleanToken) {
        return { success: false, error: 'Please enter the verification code sent to your email.' };
    }

    try {
        // Verify OTP for signup confirmation
        let res = await client.auth.verifyOtp({
            email: cleanEmail,
            token: cleanToken,
            type: 'signup'
        });

        // Fallback check with 'email' type if 'signup' type returns expired/invalid
        if (res.error && (res.error.message.includes('expired') || res.error.status === 403)) {
            const fallbackRes = await client.auth.verifyOtp({
                email: cleanEmail,
                token: cleanToken,
                type: 'email'
            });
            if (!fallbackRes.error && fallbackRes.data) {
                res = fallbackRes;
            }
        }

        if (res.error) {
            console.error('[SANADA Supabase] verifyOtp error:', res.error);
            let friendly = res.error.message;
            const errLower = (res.error.message || '').toLowerCase();
            if (errLower.includes('expired') || errLower.includes('invalid') || res.error.status === 403) {
                friendly = 'Invalid or expired 6-digit code. Please verify the code in your email or click Resend.';
            }
            return {
                success: false,
                error: friendly
            };
        }

        const user = res.data.user;
        const resolvedUsername = username || (user?.user_metadata?.username) || cleanEmail.split('@')[0];

        // Sync into public.app_users via RPC if available
        if (password) {
            try {
                const regRes = await client.rpc('register_verified_user', {
                    p_username: resolvedUsername,
                    p_password: password,
                    p_email: cleanEmail
                });
                if (regRes && regRes.data && regRes.data.success === false) {
                    return {
                        success: false,
                        error: regRes.data.error || 'Username is already taken. Please choose another username.'
                    };
                }
            } catch (rpcErr) {
                console.info('[SANADA Supabase] register_verified_user notice:', rpcErr?.message || rpcErr);
            }
        }

        return {
            success: true,
            user: {
                id: user?.id || null,
                username: resolvedUsername,
                email: cleanEmail,
                role: 'user'
            },
            session: res.data.session
        };
    } catch (err) {
        console.error('[SANADA Supabase] Unexpected verifyOtp error:', err);
        return {
            success: false,
            error: err.message || 'Verification failed. Please try again.'
        };
    }
}

/**
 * Resends the signup OTP verification code to the specified email address.
 *
 * @param {string} email - Email address to resend OTP to
 * @returns {Promise<{ success: boolean, error?: string }>}
 */
export async function resendOtpWithSupabase(email) {
    const client = getSupabaseClient();
    if (!client) {
        return { success: false, error: 'Supabase credentials are not configured.' };
    }

    const cleanEmail = (email || '').trim().toLowerCase();
    if (!cleanEmail) {
        return { success: false, error: 'Missing email address.' };
    }

    try {
        const { data, error } = await client.auth.resend({
            type: 'signup',
            email: cleanEmail
        });

        if (error) {
            console.error('[SANADA Supabase] Resend error:', error);
            let msg = error.message;
            if (error.status === 429 || error.message.toLowerCase().includes('rate limit') || error.message.toLowerCase().includes('security purposes')) {
                msg = error.message.includes('after') ? error.message : 'Please wait a short moment before requesting another code.';
            }
            return { success: false, error: msg };
        }

        return { success: true };
    } catch (err) {
        return { success: false, error: err.message || 'Failed to resend verification code.' };
    }
}

/**
 * Changes user password via PostgreSQL RPC change_password.
 *
 * @param {string} userIdOrUsername - User UUID or username
 * @param {string} oldPassword - Current plain-text password
 * @param {string} newPassword - New plain-text password (min 8 chars)
 * @returns {Promise<{ success: boolean, message?: string, error?: string }>}
 */
export async function changePasswordWithSupabase(userIdOrUsername, oldPassword, newPassword) {
    const client = getSupabaseClient();
    if (!client) {
        return { success: false, error: 'Supabase credentials are not configured.' };
    }
    const cleanOld = (oldPassword || '').trim();
    const cleanNew = (newPassword || '').trim();

    if (!cleanOld || !cleanNew) {
        return { success: false, error: 'Both current and new passwords are required.' };
    }
    if (cleanNew.length < 8) {
        return { success: false, error: 'New password must be at least 8 characters long.' };
    }
    if (cleanOld === cleanNew) {
        return { success: false, error: 'New password must be different from current password.' };
    }

    try {
        const { data, error } = await client.rpc('change_password', {
            p_user_id: String(userIdOrUsername || ''),
            p_old: cleanOld,
            p_new: cleanNew
        });

        if (error) {
            console.error('[SANADA Supabase] change_password error:', error);
            return { success: false, error: error.message || 'Database error while updating password.' };
        }

        if (data && data.success === false) {
            return { success: false, error: data.error || 'Password update failed.' };
        }

        // Keep Supabase Auth session synchronized if authenticated
        try {
            await client.auth.updateUser({ password: cleanNew });
        } catch (authErr) {}

        return { success: true, message: 'Password updated successfully!' };
    } catch (err) {
        return { success: false, error: err?.message || 'Network error updating password.' };
    }
}

/**
 * Permanently deletes the user account, verifying credentials via password.
 * Removes user records from public.app_users, api_keys, run_history, and auth.users.
 *
 * @param {string} userIdOrUsername - User UUID or username
 * @param {string} password - Current plain-text password for confirmation
 * @returns {Promise<{ success: boolean, message?: string, error?: string }>}
 */
export async function deleteUserAccountWithSupabase(userIdOrUsername, password) {
    const client = getSupabaseClient();
    if (!client) {
        return { success: false, error: 'Supabase credentials are not configured.' };
    }

    const cleanPass = (password || '').trim();
    if (!cleanPass) {
        return { success: false, error: 'Please enter your password to confirm account deletion.' };
    }

    const userIdent = String(userIdOrUsername || '').trim();
    if (!userIdent) {
        return { success: false, error: 'User identifier not provided.' };
    }

    const DEMO_NAMES = ['developer', 'user', 'admin'];
    if (DEMO_NAMES.includes(userIdent.toLowerCase())) {
        return { success: false, error: 'Built-in demo accounts cannot be deleted.' };
    }

    try {
        // 1. Try PostgreSQL delete_user_account RPC
        const { data, error } = await client.rpc('delete_user_account', {
            p_user_id: userIdent,
            p_password: cleanPass
        });

        if (!error && data) {
            if (data.success === false) {
                return { success: false, error: data.error || 'Account deletion failed.' };
            }
            return { success: true, message: data.message || 'Account deleted successfully.' };
        }

        // 2. Fallback: verify password via authenticate_user RPC then delete records
        const authCheck = await client.rpc('authenticate_user', {
            p_username: userIdent,
            p_password: cleanPass
        });

        if (authCheck.error || !Array.isArray(authCheck.data) || authCheck.data.length === 0) {
            return { success: false, error: 'Incorrect password. Account deletion aborted.' };
        }

        // Clean up linked data
        try {
            await client.from('api_keys').delete().ilike('user_identifier', userIdent);
            await client.from('run_history').delete().ilike('user_identifier', userIdent);
        } catch (subErr) {}

        const { error: delErr } = await client
            .from('app_users')
            .delete()
            .ilike('username', userIdent);

        if (delErr) {
            return { success: false, error: delErr.message || 'Failed to delete user account.' };
        }

        return { success: true, message: 'Account deleted successfully.' };
    } catch (err) {
        console.error('[SANADA Supabase] deleteUserAccount error:', err);
        return { success: false, error: err?.message || 'Network error while attempting account deletion.' };
    }
}

// =============================================================================
// 4. CONSOLE CONFIGURATION HELPER
// =============================================================================
// Allows setting or updating credentials instantly in devtools without touching files:
// window.setSupabaseConfig('https://xyz.supabase.co', 'eyJh...')
if (typeof window !== 'undefined') {
    window.setSupabaseConfig = function(url, anonKey) {
        if (!url || !anonKey) {
            console.error('[SANADA] Usage: setSupabaseConfig("https://xyz.supabase.co", "eyJhbGciOi...")');
            return;
        }
        localStorage.setItem('sanada_supabase_url', url.trim());
        localStorage.setItem('sanada_supabase_anon_key', anonKey.trim());
        console.log('[SANADA] Supabase credentials saved to localStorage!');
        console.log('[SANADA] Reloading to apply credentials...');
        window.location.reload();
    };

    window.SANADA_SUPABASE = {
        isConfigured: isSupabaseConfigured,
        getClient: getSupabaseClient,
        login: loginWithSupabase,
        signUp: signUpWithSupabase,
        verifyOtp: verifyOtpWithSupabase,
        resendOtp: resendOtpWithSupabase,
        changePassword: changePasswordWithSupabase,
        deleteAccount: deleteUserAccountWithSupabase,
        getSupabaseUrl,
        getSupabaseKey
    };
}
