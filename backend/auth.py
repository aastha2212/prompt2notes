"""
Authentication module for Prompt2Notes.
Handles user registration, login, and session management.
"""

import os
import json
import hashlib
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Default users file path
USERS_FILE = Path("users.json")


class AuthManager:
    """
    Manages user authentication, registration, and session handling.
    """
    
    def __init__(self, users_file: Optional[Path] = None):
        """
        Initialize AuthManager.
        
        Args:
            users_file: Path to JSON file storing user credentials
        """
        self.users_file = users_file or USERS_FILE
        self._ensure_users_file()
        self._load_users()
    
    def _ensure_users_file(self):
        """Create users file if it doesn't exist."""
        if not self.users_file.exists():
            # Create default admin user (username: admin, password: admin)
            default_users = {
                "admin": {
                    "password_hash": self._hash_password("admin"),
                    "email": "admin@prompt2notes.com",
                    "created_at": "2024-01-01"
                }
            }
            self._save_users(default_users)
            logger.info(f"Created default users file at {self.users_file}")
            logger.warning("Default admin user created: username='admin', password='admin' - Please change this!")
    
    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """Load users from JSON file."""
        try:
            if self.users_file.exists():
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load users: {e}")
            return {}
    
    def _save_users(self, users: Dict[str, Dict[str, Any]]):
        """Save users to JSON file."""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save users: {e}")
            raise
    
    def _hash_password(self, password: str) -> str:
        """
        Hash password using SHA-256 (simple hashing for MVP).
        For production, use bcrypt or argon2.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password
            password_hash: Stored password hash
            
        Returns:
            True if password matches
        """
        return self._hash_password(password) == password_hash
    
    def register_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Register a new user.
        
        Args:
            username: Username (must be unique)
            password: Plain text password
            email: Optional email address
            
        Returns:
            Tuple of (success, message)
        """
        if not username or not password:
            return False, "Username and password are required"
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if len(password) < 4:
            return False, "Password must be at least 4 characters"
        
        users = self._load_users()
        
        if username in users:
            return False, "Username already exists"
        
        users[username] = {
            "password_hash": self._hash_password(password),
            "email": email or "",
            "created_at": str(Path(__file__).stat().st_mtime)  # Simple timestamp
        }
        
        try:
            self._save_users(users)
            logger.info(f"User registered: {username}")
            return True, "User registered successfully"
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False, f"Registration failed: {e}"
    
    def authenticate(self, username: str, password: str) -> tuple[bool, Optional[str]]:
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            Tuple of (success, error_message)
        """
        users = self._load_users()
        
        if username not in users:
            return False, "Invalid username or password"
        
        user = users[username]
        password_hash = user.get("password_hash")
        
        if not self.verify_password(password, password_hash):
            return False, "Invalid username or password"
        
        logger.info(f"User authenticated: {username}")
        return True, None
    
    def user_exists(self, username: str) -> bool:
        """Check if user exists."""
        users = self._load_users()
        return username in users
    
    def change_password(
        self,
        username: str,
        old_password: str,
        new_password: str
    ) -> tuple[bool, str]:
        """
        Change user password.
        
        Args:
            username: Username
            old_password: Current password
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        users = self._load_users()
        
        if username not in users:
            return False, "User not found"
        
        user = users[username]
        
        # Verify old password
        if not self.verify_password(old_password, user["password_hash"]):
            return False, "Current password is incorrect"
        
        if len(new_password) < 4:
            return False, "New password must be at least 4 characters"
        
        # Update password
        user["password_hash"] = self._hash_password(new_password)
        
        try:
            self._save_users(users)
            logger.info(f"Password changed for user: {username}")
            return True, "Password changed successfully"
        except Exception as e:
            logger.error(f"Password change failed: {e}")
            return False, f"Password change failed: {e}"


def get_auth_manager() -> AuthManager:
    """Get singleton AuthManager instance."""
    if not hasattr(get_auth_manager, '_instance'):
        get_auth_manager._instance = AuthManager()
    return get_auth_manager._instance

