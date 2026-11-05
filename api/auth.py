"""API Authentication and Tenant Context Management."""

from __future__ import annotations
import os
import secrets
from typing import Dict, Optional, Set
from fastapi import Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel


class TenantContext(BaseModel):
    """Context identifying the authenticated tenant and permissions."""
    tenant_id: str
    user_id: str
    is_admin: bool = False


# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

DEFAULT_DEV_API_KEY = os.getenv("API_KEY", "rag-opt-lab-dev-key-2025")


def get_configured_api_keys() -> Dict[str, str]:
    """Retrieve mapping of valid API keys to tenant IDs.
    
    Format in env: API_KEYS="key_alpha:tenant_a,key_beta:tenant_b,master_key:admin"
    Or single default: API_KEY="my-secret-key"
    """
    keys_env = os.getenv("API_KEYS", "")
    key_mapping: Dict[str, str] = {}

    if keys_env.strip():
        for pair in keys_env.split(","):
            if ":" in pair:
                k, t = pair.split(":", 1)
                key_mapping[k.strip()] = t.strip()
            elif pair.strip():
                key_mapping[pair.strip()] = "default"
    else:
        # Default development key
        key_mapping[DEFAULT_DEV_API_KEY] = "default"

    return key_mapping


async def get_current_tenant(
    api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
) -> TenantContext:
    """Validate API credentials and resolve isolated tenant identity.
    
    Accepts:
    - X-API-Key header
    - Authorization: Bearer <key>
    - Optional X-Tenant-ID header to scope multi-tenant access
    """
    token = api_key or (bearer.credentials if bearer else None)
    valid_keys = get_configured_api_keys()
    allow_anonymous = os.getenv("ALLOW_ANONYMOUS", "false").lower() in {"1", "true", "yes"}

    if not token:
        if allow_anonymous:
            resolved_tenant = x_tenant_id or "default"
            return TenantContext(tenant_id=resolved_tenant, user_id=f"anon-{resolved_tenant}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Provide a valid API key via 'X-API-Key' or 'Authorization: Bearer <token>' header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    # Secure constant-time comparison
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

    # If the key is an admin key or user provides specific tenant header
    is_admin = (matched_tenant == "admin")
    if is_admin and x_tenant_id:
        active_tenant = x_tenant_id
    elif x_tenant_id and x_tenant_id == matched_tenant:
        active_tenant = x_tenant_id
    elif matched_tenant == "default" and x_tenant_id:
        active_tenant = x_tenant_id
    else:
        active_tenant = matched_tenant

    # Sanitize tenant identifier to prevent path traversal in tenant folder creation
    safe_tenant_id = "".join(c for c in active_tenant if c.isalnum() or c in {"-", "_"}).strip()
    if not safe_tenant_id:
        safe_tenant_id = "default"

    return TenantContext(
        tenant_id=safe_tenant_id,
        user_id=f"user-{safe_tenant_id}",
        is_admin=is_admin
    )

