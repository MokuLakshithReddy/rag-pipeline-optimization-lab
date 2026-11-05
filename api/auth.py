"""API Authentication and Tenant Context Management.

Provides strict fail-closed authentication, tenant-scoped credential resolution,
and enforced cross-tenant boundary protections.
"""

from __future__ import annotations
import logging
import os
import secrets
from typing import Dict, Optional
from fastapi import Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logger = logging.getLogger("api_auth")


class TenantContext(BaseModel):
    """Context identifying the authenticated tenant, user, and administrative privileges."""
    tenant_id: str
    user_id: str
    is_admin: bool = False


# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def get_configured_api_keys() -> Dict[str, str]:
    """Retrieve mapping of valid API keys to tenant IDs.
    
    Format in env:
      API_KEYS="key_alpha:tenant_a,key_beta:tenant_b,master_admin_key:admin"
    Or single tenant key in env:
      API_KEY="my-secret-key" (binds strictly to 'default' tenant)
    
    If neither API_KEYS nor API_KEY is set in environment, returns empty dict (fail-closed).
    """
    keys_env = os.getenv("API_KEYS", "")
    single_key = os.getenv("API_KEY", "")
    key_mapping: Dict[str, str] = {}

    if keys_env.strip():
        for pair in keys_env.split(","):
            if ":" in pair:
                k, t = pair.split(":", 1)
                if k.strip() and t.strip():
                    key_mapping[k.strip()] = t.strip()
            elif pair.strip():
                key_mapping[pair.strip()] = "default"
    elif single_key.strip():
        key_mapping[single_key.strip()] = "default"

    return key_mapping


async def get_current_tenant(
    api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
) -> TenantContext:
    """Validate API credentials and resolve isolated tenant identity with strict authorization.
    
    Security Guarantees:
    1. Fail-closed: Rejects all requests if no credentials are provided or configured.
    2. Tenant Binding: Each API key is strictly bound to its assigned tenant namespace.
    3. Cross-Tenant Escalation Prevention: Ordinary tenants attempting to choose another
       tenant via 'X-Tenant-ID' receive HTTP 403 Forbidden.
    4. Administrative Privileges: Cross-tenant switching is exclusively permitted for
       keys explicitly mapped to the 'admin' role.
    """
    token = api_key or (bearer.credentials if bearer else None)
    valid_keys = get_configured_api_keys()
    allow_anonymous = os.getenv("ALLOW_ANONYMOUS", "false").lower() in {"1", "true", "yes"}

    # 1. Anonymous Access Handling (disabled by default in production)
    if not token:
        if allow_anonymous:
            resolved_tenant = "default"
            return TenantContext(tenant_id=resolved_tenant, user_id=f"anon-{resolved_tenant}", is_admin=False)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Provide a valid API key via 'X-API-Key' or 'Authorization: Bearer <token>' header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    # 2. Fail-closed check if no keys are configured
    if not valid_keys:
        logger.error("Authentication rejected: No API_KEYS or API_KEY configured in environment.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication service unconfigured: No valid API keys are registered in the server environment.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    # 3. Secure constant-time token comparison
    matched_tenant = None
    for valid_key, tenant in valid_keys.items():
        if secrets.compare_digest(token, valid_key):
            matched_tenant = tenant
            break

    if not matched_tenant:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key: Access denied.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    # 4. Strict Role & Cross-Tenant Boundary Enforcement
    is_admin = (matched_tenant == "admin")

    if x_tenant_id and x_tenant_id.strip():
        requested_tenant = x_tenant_id.strip()
        if is_admin:
            # Administrators are explicitly authorized to switch tenant workspaces
            active_tenant = requested_tenant
        else:
            # Ordinary tenant attempting cross-tenant access
            if requested_tenant != matched_tenant:
                logger.warning(
                    f"Cross-tenant access blocked: tenant '{matched_tenant}' attempted to access "
                    f"workspace '{requested_tenant}' via X-Tenant-ID header."
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: Ordinary tenants are not authorized to switch tenant workspaces. "
                           f"Your key is bound exclusively to tenant '{matched_tenant}'."
                )
            active_tenant = matched_tenant
    else:
        active_tenant = matched_tenant

    # 5. Sanitize tenant identifier to prevent directory traversal
    safe_tenant_id = "".join(c for c in active_tenant if c.isalnum() or c in {"-", "_"}).strip()
    if not safe_tenant_id:
        safe_tenant_id = "default"

    return TenantContext(
        tenant_id=safe_tenant_id,
        user_id=f"user-{safe_tenant_id}",
        is_admin=is_admin
    )
