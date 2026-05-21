"""
System Health Check
Comprehensive health monitoring for the trading bot.
Checks dependencies, configuration, APIs, and system resources.
"""

import sys
import os
import platform
import subprocess
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import importlib.util


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Single health check result"""
    component: str
    status: HealthStatus
    message: str
    details: str = ""

    def __str__(self) -> str:
        icon = {
            HealthStatus.HEALTHY: "OK",
            HealthStatus.WARNING: "WARN",
            HealthStatus.CRITICAL: "FAIL",
            HealthStatus.UNKNOWN: "?"
        }[self.status]
        
        msg = f"{icon} {self.component}: {self.message}"
        if self.details:
            msg += f"\n   {self.details}"
        return msg


class SystemHealthChecker:
    """Comprehensive system health checker"""

    def __init__(self, config: Dict):
        self.config = config
        self.checks: List[HealthCheck] = []

    def check_all(self) -> Tuple[bool, List[HealthCheck]]:
        """
        Run all health checks.
        
        Returns:
            (is_healthy, checks) - True if no critical issues, list of all checks
        """
        self.checks = []
        
        # System checks
        self._check_python_version()
        self._check_platform()
        self._check_disk_space()
        
        # Dependency checks
        self._check_required_packages()
        self._check_optional_packages()
        
        # Configuration checks
        self._check_config_files()
        self._check_directories()
        
        # API checks (quick connectivity, not full validation)
        self._check_internet_connectivity()
        
        # Resource checks
        self._check_memory()
        
        # Determine overall health
        has_critical = any(c.status == HealthStatus.CRITICAL for c in self.checks)
        
        return (not has_critical, self.checks)

    def _check_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        current = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major < 3:
            self.checks.append(HealthCheck(
                component="Python Version",
                status=HealthStatus.CRITICAL,
                message=f"Python {current} - requires Python 3.8+",
                details="Please upgrade to Python 3.8 or higher"
            ))
        elif version.minor < 8:
            self.checks.append(HealthCheck(
                component="Python Version",
                status=HealthStatus.CRITICAL,
                message=f"Python {current} - requires Python 3.8+",
                details="Please upgrade to Python 3.8 or higher"
            ))
        elif version.minor < 10:
            self.checks.append(HealthCheck(
                component="Python Version",
                status=HealthStatus.WARNING,
                message=f"Python {current} - consider upgrading to 3.10+",
                details="Some features may work better on Python 3.10+"
            ))
        else:
            self.checks.append(HealthCheck(
                component="Python Version",
                status=HealthStatus.HEALTHY,
                message=f"Python {current}",
                details="Version compatible"
            ))

    def _check_platform(self):
        """Check operating system"""
        system = platform.system()
        release = platform.release()
        
        self.checks.append(HealthCheck(
            component="Operating System",
            status=HealthStatus.HEALTHY,
            message=f"{system} {release}",
            details=f"Architecture: {platform.machine()}"
        ))

    def _check_disk_space(self):
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            
            free_gb = free // (2**30)
            free_pct = (free / total) * 100
            
            if free_gb < 1:
                status = HealthStatus.CRITICAL
                msg = f"{free_gb}GB free - critically low"
                details = "Need at least 1GB for logs and data"
            elif free_gb < 5:
                status = HealthStatus.WARNING
                msg = f"{free_gb}GB free - running low"
                details = "Consider freeing up space"
            else:
                status = HealthStatus.HEALTHY
                msg = f"{free_gb}GB free ({free_pct:.1f}%)"
                details = "Sufficient disk space available"
            
            self.checks.append(HealthCheck(
                component="Disk Space",
                status=status,
                message=msg,
                details=details
            ))
        except Exception as e:
            self.checks.append(HealthCheck(
                component="Disk Space",
                status=HealthStatus.UNKNOWN,
                message="Could not check disk space",
                details=str(e)[:100]
            ))

    def _check_required_packages(self):
        """Check required Python packages"""
        required = [
            ("pandas", "Data manipulation"),
            ("numpy", "Numerical computing"),
            ("ccxt", "Crypto exchange connectivity"),
            ("anthropic", "Claude LLM integration"),
            ("dotenv", "Environment variables"),
            ("streamlit", "Dashboard UI"),
        ]
        
        missing = []
        outdated = []
        
        for package, description in required:
            # Check if package is installed
            spec = importlib.util.find_spec(package.replace("-", "_"))
            
            if spec is None:
                missing.append(f"{package} ({description})")
        
        if missing:
            self.checks.append(HealthCheck(
                component="Required Packages",
                status=HealthStatus.CRITICAL,
                message=f"{len(missing)} required packages missing",
                details=f"Missing: {', '.join(missing)}\nRun: pip install " + " ".join([p.split()[0] for p in missing])
            ))
        else:
            self.checks.append(HealthCheck(
                component="Required Packages",
                status=HealthStatus.HEALTHY,
                message="All required packages installed",
                details=f"Checked {len(required)} packages"
            ))

    def _check_optional_packages(self):
        """Check optional Python packages"""
        optional = [
            ("ta", "Technical analysis indicators"),
            ("yfinance", "Stock data retrieval"),
            ("plotly", "Advanced charting"),
            ("feedparser", "RSS feed parsing"),
            ("beautifulsoup4", "Web scraping"),
        ]
        
        missing = []
        
        for package, description in optional:
            spec = importlib.util.find_spec(package.replace("-", "_"))
            if spec is None:
                missing.append(f"{package} ({description})")
        
        if missing:
            self.checks.append(HealthCheck(
                component="Optional Packages",
                status=HealthStatus.WARNING,
                message=f"{len(missing)} optional packages missing",
                details=f"Missing: {', '.join(missing)}\nThese enhance functionality but are not required"
            ))
        else:
            self.checks.append(HealthCheck(
                component="Optional Packages",
                status=HealthStatus.HEALTHY,
                message="All optional packages installed",
                details=f"Checked {len(optional)} packages"
            ))

    def _check_config_files(self):
        """Check configuration files exist and are valid"""
        config_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(config_dir)
        
        # Check .env file
        env_path = os.path.join(project_root, ".env")
        if not os.path.exists(env_path):
            self.checks.append(HealthCheck(
                component="Config Files",
                status=HealthStatus.WARNING,
                message=".env file not found",
                details=f"Create {env_path} from .env.example template"
            ))
        else:
            # Check if .env has any real values (not just placeholders)
            with open(env_path, 'r') as f:
                content = f.read()
            
            has_real_values = any(
                line.strip() and '=' in line and 
                not line.strip().startswith('#') and
                'your_' not in line.lower() and
                line.split('=')[1].strip() != ''
                for line in content.split('\n')
            )
            
            if has_real_values:
                self.checks.append(HealthCheck(
                    component="Config Files",
                    status=HealthStatus.HEALTHY,
                    message=".env file configured",
                    details="Environment variables loaded"
                ))
            else:
                self.checks.append(HealthCheck(
                    component="Config Files",
                    status=HealthStatus.WARNING,
                    message=".env file exists but not configured",
                    details="Update .env with your API keys and settings"
                ))

    def _check_directories(self):
        """Check required directories exist"""
        config_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(config_dir)
        
        required_dirs = [
            "logs",
            "models/saved",
            "data/historical",
        ]
        
        missing_dirs = []
        created_dirs = []
        
        for dir_path in required_dirs:
            full_path = os.path.join(project_root, dir_path)
            if not os.path.exists(full_path):
                try:
                    os.makedirs(full_path, exist_ok=True)
                    created_dirs.append(dir_path)
                except Exception as e:
                    missing_dirs.append(f"{dir_path} ({str(e)})")
        
        if missing_dirs:
            self.checks.append(HealthCheck(
                component="Directories",
                status=HealthStatus.CRITICAL,
                message=f"Could not create {len(missing_dirs)} required directories",
                details=f"Failed: {', '.join(missing_dirs)}"
            ))
        elif created_dirs:
            self.checks.append(HealthCheck(
                component="Directories",
                status=HealthStatus.HEALTHY,
                message="Required directories initialized",
                details=f"Created: {', '.join(created_dirs)}"
            ))
        else:
            self.checks.append(HealthCheck(
                component="Directories",
                status=HealthStatus.HEALTHY,
                message="All required directories exist",
                details=f"Checked {len(required_dirs)} directories"
            ))

    def _check_internet_connectivity(self):
        """Check basic internet connectivity"""
        try:
            import socket
            
            # Try to resolve google.com
            socket.setdefaulttimeout(3)
            socket.gethostbyname("google.com")
            
            self.checks.append(HealthCheck(
                component="Internet",
                status=HealthStatus.HEALTHY,
                message="Internet connection active",
                details="DNS resolution working"
            ))
        except Exception as e:
            self.checks.append(HealthCheck(
                component="Internet",
                status=HealthStatus.CRITICAL,
                message="No internet connection",
                details="Required for API access and data fetching"
            ))

    def _check_memory(self):
        """Check available memory"""
        try:
            # Try psutil first (if available)
            try:
                import psutil
                mem = psutil.virtual_memory()
                available_mb = mem.available // (1024 * 1024)
                percent_used = mem.percent
                
                if available_mb < 256:
                    status = HealthStatus.CRITICAL
                    msg = f"{available_mb}MB available - critically low"
                    details = "Need at least 256MB for stable operation"
                elif available_mb < 512:
                    status = HealthStatus.WARNING
                    msg = f"{available_mb}MB available - running low"
                    details = "Consider closing other applications"
                else:
                    status = HealthStatus.HEALTHY
                    msg = f"{available_mb}MB available ({100-percent_used:.1f}% free)"
                    details = "Sufficient memory available"
                
                self.checks.append(HealthCheck(
                    component="Memory",
                    status=status,
                    message=msg,
                    details=details
                ))
            except ImportError:
                # psutil not available, skip memory check
                self.checks.append(HealthCheck(
                    component="Memory",
                    status=HealthStatus.UNKNOWN,
                    message="Cannot check memory (psutil not installed)",
                    details="Optional: pip install psutil for memory monitoring"
                ))
        except Exception as e:
            self.checks.append(HealthCheck(
                component="Memory",
                status=HealthStatus.UNKNOWN,
                message="Could not check memory",
                details=str(e)[:100]
            ))

    def get_summary(self) -> str:
        """Generate health check summary"""
        if not self.checks:
            return "No health checks run yet"
        
        lines = []
        lines.append("=" * 70)
        lines.append("SYSTEM HEALTH CHECK")
        lines.append("=" * 70)
        lines.append("")
        
        # Count statuses
        healthy = sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY)
        warning = sum(1 for c in self.checks if c.status == HealthStatus.WARNING)
        critical = sum(1 for c in self.checks if c.status == HealthStatus.CRITICAL)
        unknown = sum(1 for c in self.checks if c.status == HealthStatus.UNKNOWN)
        
        total = len(self.checks)
        lines.append(f"Healthy: {healthy}/{total}  |  Warnings: {warning}  |  Critical: {critical}  |  Unknown: {unknown}")
        lines.append("")
        
        # Overall status
        if critical > 0:
            lines.append("OVERALL STATUS: CRITICAL - Bot may not function properly")
        elif warning > 0:
            lines.append("OVERALL STATUS: WARNING - Some features may be limited")
        else:
            lines.append("OVERALL STATUS: HEALTHY - All systems operational")
        
        lines.append("")
        lines.append("-" * 70)
        lines.append("")
        
        # Show each check
        for check in self.checks:
            lines.append(str(check))
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)

    def get_quick_status(self) -> str:
        """Get one-line status summary"""
        if not self.checks:
            return "? No checks run"
        
        healthy = sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY)
        warning = sum(1 for c in self.checks if c.status == HealthStatus.WARNING)
        critical = sum(1 for c in self.checks if c.status == HealthStatus.CRITICAL)
        
        if critical > 0:
            return f"CRITICAL: {critical} issues"
        elif warning > 0:
            return f"WARNING: {warning} issues"
        else:
            return f"HEALTHY: {healthy}/{len(self.checks)} checks passed"


def check_system_health(config: Dict, verbose: bool = True) -> bool:
    """
    Convenience function to check system health.
    
    Args:
        config: Configuration dictionary
        verbose: Print summary if True
        
    Returns:
        True if healthy (no critical issues), False otherwise
    """
    checker = SystemHealthChecker(config)
    is_healthy, checks = checker.check_all()
    
    if verbose:
        print(checker.get_summary())
    
    return is_healthy


if __name__ == "__main__":
    # Test with current config
    from config import CONFIG
    check_system_health(CONFIG)
