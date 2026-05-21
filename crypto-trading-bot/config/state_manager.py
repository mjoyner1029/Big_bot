"""State Manager — Atomic writes and automatic backups for bot state.

Prevents state corruption and data loss. Inspired by the Meteora LP bot's
recovery tooling that saved the day when v1 state was corrupted.

Features:
  • Atomic writes (tmp file → rename) prevent partial writes
  • Timestamped backups kept for configurable retention period
  • Automatic pruning of old backups
  • Corruption detection and recovery
  • State validation

Usage:
    from config.state_manager import StateManager
    
    manager = StateManager("portfolio.json")
    
    # Save state (automatically creates backup)
    manager.save({"positions": [...], "balance": 1000})
    
    # Load state (recovers from backup if corrupted)
    state = manager.load()
"""
import logging
import os
import json
import shutil
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
from pathlib import Path


class StateManager:
    """Manages persistent state with atomic writes and backups."""
    
    def __init__(
        self,
        filename: str,
        state_dir: str = "state",
        backup_retention_hours: int = 72,
        auto_backup: bool = True,
    ):
        """Initialize state manager.
        
        Args:
            filename: Name of the state file (e.g., "portfolio.json")
            state_dir: Directory for state files
            backup_retention_hours: How long to keep backups
            auto_backup: Whether to create timestamped backups on save
        """
        self.filename = filename
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / filename
        self.backup_dir = self.state_dir / "backups" / filename.replace(".json", "")
        self.backup_retention_hours = backup_retention_hours
        self.auto_backup = auto_backup
        
        # Create directories
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if auto_backup:
            self.backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    
    def save(self, data: Dict[str, Any], create_backup: bool = None) -> bool:
        """Save state with atomic write and optional backup.
        
        Args:
            data: State data to save
            create_backup: Override auto_backup setting for this save
        
        Returns:
            True if successful, False otherwise
        """
        if create_backup is None:
            create_backup = self.auto_backup
        
        try:
            # Add metadata
            data_with_meta = {
                **data,
                "_metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "version": data.get("_metadata", {}).get("version", 1),
                }
            }
            
            # Atomic write: write to temp file then rename
            tmp_file = self.state_file.with_suffix(".tmp")
            
            with open(tmp_file, "w") as f:
                json.dump(data_with_meta, f, indent=2)
            
            # Atomic rename (overwrites existing file)
            tmp_file.replace(self.state_file)
            
            # Create timestamped backup
            if create_backup:
                self._create_backup()
            
            logging.debug(f"[StateManager] Saved {self.filename}")
            return True
            
        except Exception as e:
            logging.error(f"[StateManager] Failed to save {self.filename}: {e}")
            return False
    
    def load(self, validate: bool = True) -> Optional[Dict[str, Any]]:
        """Load state from file, with automatic recovery from backup if corrupted.
        
        Args:
            validate: Whether to validate loaded data
        
        Returns:
            State dict, or None if file doesn't exist
        """
        if not self.state_file.exists():
            logging.info(f"[StateManager] No state file found: {self.filename}")
            return None
        
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            
            if validate and not self._validate_state(data):
                raise ValueError("State validation failed")
            
            logging.debug(f"[StateManager] Loaded {self.filename}")
            return data
            
        except Exception as e:
            logging.error(
                f"[StateManager] Failed to load {self.filename}: {e}. "
                f"Attempting recovery from backup..."
            )
            
            # Try to recover from backup
            recovered = self._recover_from_backup()
            if recovered is not None:
                logging.warning(
                    f"[StateManager] Successfully recovered {self.filename} "
                    f"from backup"
                )
                # Save the recovered state as current
                self.save(recovered, create_backup=False)
                return recovered
            
            logging.error(
                f"[StateManager] Could not recover {self.filename}. "
                f"Returning None."
            )
            return None
    
    def _create_backup(self) -> None:
        """Create timestamped backup of current state file."""
        if not self.state_file.exists():
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_file = self.backup_dir / f"{timestamp}.json"
            
            shutil.copy2(self.state_file, backup_file)
            
            # Prune old backups
            self._prune_backups()
            
            logging.debug(f"[StateManager] Created backup: {backup_file.name}")
            
        except Exception as e:
            logging.warning(f"[StateManager] Failed to create backup: {e}")
    
    def _prune_backups(self) -> None:
        """Remove backups older than retention period."""
        if not self.backup_dir.exists():
            return
        
        try:
            cutoff = datetime.now() - timedelta(hours=self.backup_retention_hours)
            
            for backup_file in self.backup_dir.glob("*.json"):
                try:
                    mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if mtime < cutoff:
                        backup_file.unlink()
                        logging.debug(
                            f"[StateManager] Pruned old backup: {backup_file.name}"
                        )
                except Exception as e:
                    logging.warning(
                        f"[StateManager] Failed to prune backup {backup_file.name}: {e}"
                    )
            
        except Exception as e:
            logging.warning(f"[StateManager] Failed to prune backups: {e}")
    
    def _recover_from_backup(self) -> Optional[Dict[str, Any]]:
        """Try to recover state from most recent valid backup."""
        if not self.backup_dir.exists():
            return None
        
        # Get all backups sorted by modification time (newest first)
        backups = sorted(
            self.backup_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for backup_file in backups:
            try:
                with open(backup_file, "r") as f:
                    data = json.load(f)
                
                if self._validate_state(data):
                    logging.info(
                        f"[StateManager] Recovered from backup: {backup_file.name}"
                    )
                    return data
                    
            except Exception as e:
                logging.debug(
                    f"[StateManager] Backup {backup_file.name} is invalid: {e}"
                )
                continue
        
        return None
    
    def _validate_state(self, data: Dict[str, Any]) -> bool:
        """Validate state data structure.
        
        Override this method in subclasses for custom validation.
        """
        # Basic validation: must be a dict
        if not isinstance(data, dict):
            return False
        
        # Check for metadata (optional but recommended)
        if "_metadata" in data:
            meta = data["_metadata"]
            if not isinstance(meta, dict):
                return False
            
            # Check for required metadata fields
            if "last_updated" not in meta:
                return False
        
        return True
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with metadata.
        
        Returns:
            List of dicts with backup info
        """
        if not self.backup_dir.exists():
            return []
        
        backups = []
        for backup_file in sorted(
            self.backup_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        ):
            try:
                stat = backup_file.stat()
                backups.append({
                    "filename": backup_file.name,
                    "path": str(backup_file),
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "size_bytes": stat.st_size,
                })
            except Exception:
                continue
        
        return backups
    
    def restore_from_backup(self, backup_filename: str) -> bool:
        """Manually restore state from a specific backup.
        
        Args:
            backup_filename: Name of backup file (e.g., "20260507-143022.json")
        
        Returns:
            True if successful, False otherwise
        """
        backup_file = self.backup_dir / backup_filename
        
        if not backup_file.exists():
            logging.error(f"[StateManager] Backup not found: {backup_filename}")
            return False
        
        try:
            with open(backup_file, "r") as f:
                data = json.load(f)
            
            if not self._validate_state(data):
                logging.error(f"[StateManager] Backup validation failed: {backup_filename}")
                return False
            
            # Save as current state
            if self.save(data):
                logging.info(
                    f"[StateManager] Restored from backup: {backup_filename}"
                )
                return True
            
            return False
            
        except Exception as e:
            logging.error(
                f"[StateManager] Failed to restore from backup {backup_filename}: {e}"
            )
            return False
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get information about current state file.
        
        Returns:
            Dict with state file info
        """
        if not self.state_file.exists():
            return {
                "exists": False,
                "filename": self.filename,
                "path": str(self.state_file),
            }
        
        try:
            stat = self.state_file.stat()
            return {
                "exists": True,
                "filename": self.filename,
                "path": str(self.state_file),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "backups_count": len(list(self.backup_dir.glob("*.json")))
                    if self.backup_dir.exists() else 0,
            }
        except Exception as e:
            return {
                "exists": True,
                "filename": self.filename,
                "path": str(self.state_file),
                "error": str(e),
            }


# ── Convenience functions for common state files ─────────────────

def get_portfolio_state_manager() -> StateManager:
    """Get state manager for portfolio state."""
    return StateManager("portfolio.json")


def get_bot_state_manager() -> StateManager:
    """Get state manager for general bot state."""
    return StateManager("bot_state.json")


def get_state_summary() -> Dict[str, Any]:
    """Get summary of all state files (for monitoring)."""
    managers = [
        get_portfolio_state_manager(),
        get_bot_state_manager(),
        StateManager("killswitch.json"),
        StateManager("position_health.json"),
    ]
    
    summary = {
        "state_files": [],
        "total_backups": 0,
    }
    
    for manager in managers:
        info = manager.get_state_info()
        backups = manager.list_backups()
        
        info["backups"] = len(backups)
        summary["state_files"].append(info)
        summary["total_backups"] += len(backups)
    
    return summary
