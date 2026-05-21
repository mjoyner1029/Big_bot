#!/usr/bin/env python3
"""
Dependency Verification System
Checks all Python dependencies, versions, and compatibility.
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class DependencyStatus(Enum):
    """Dependency check status"""
    INSTALLED = "installed"
    MISSING = "missing"
    OUTDATED = "outdated"
    INCOMPATIBLE = "incompatible"


@dataclass
class Dependency:
    """Package dependency information"""
    name: str
    required_version: str = ""
    installed_version: str = ""
    status: DependencyStatus = DependencyStatus.MISSING
    category: str = "core"  # core, optional, dev


class DependencyChecker:
    """Comprehensive dependency verification"""

    def __init__(self):
        self.dependencies: List[Dependency] = []
        self.project_root = Path(__file__).parent.parent

    def check_all(self) -> Tuple[bool, List[Dependency]]:
        """
        Check all dependencies.
        
        Returns:
            (all_satisfied, dependencies) - True if all required deps satisfied
        """
        self._load_requirements()
        self._check_each_dependency()
        
        # Check if all core dependencies are satisfied
        core_missing = [d for d in self.dependencies 
                       if d.category == "core" and d.status == DependencyStatus.MISSING]
        
        return (len(core_missing) == 0, self.dependencies)

    def _load_requirements(self):
        """Load requirements from requirements.txt"""
        req_file = self.project_root / "requirements.txt"
        
        if not req_file.exists():
            print(f"Warning: {req_file} not found")
            # Define core requirements manually
            self._define_core_requirements()
            return
        
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse package name and version
                if '>=' in line:
                    name, version = line.split('>=')
                    required_version = f">={version}"
                elif '==' in line:
                    name, version = line.split('==')
                    required_version = f"=={version}"
                else:
                    name = line
                    required_version = ""
                
                # Determine category
                category = self._categorize_package(name.strip())
                
                self.dependencies.append(Dependency(
                    name=name.strip(),
                    required_version=required_version.strip(),
                    category=category
                ))

    def _define_core_requirements(self):
        """Define core requirements if requirements.txt missing"""
        core_packages = [
            ("pandas", "core"),
            ("numpy", "core"),
            ("scikit-learn", "core"),
            ("xgboost", "core"),
            ("ccxt", "core"),
            ("anthropic", "core"),
            ("python-dotenv", "core"),
            ("streamlit", "core"),
            ("plotly", "core"),
            ("ta", "core"),
            ("yfinance", "core"),
            ("alpaca-trade-api", "core"),
            ("requests", "core"),
            ("beautifulsoup4", "optional"),
            ("nltk", "optional"),
            ("vaderSentiment", "optional"),
            ("feedparser", "optional"),
            ("websocket-client", "optional"),
        ]
        
        for name, category in core_packages:
            self.dependencies.append(Dependency(
                name=name,
                category=category
            ))

    def _categorize_package(self, name: str) -> str:
        """Categorize package as core, optional, or dev"""
        core_packages = {
            'pandas', 'numpy', 'scikit-learn', 'xgboost', 'ccxt',
            'anthropic', 'python-dotenv', 'streamlit', 'plotly',
            'ta', 'yfinance', 'alpaca-trade-api', 'requests'
        }
        
        optional_packages = {
            'beautifulsoup4', 'lxml', 'nltk', 'vaderSentiment',
            'feedparser', 'websocket-client', 'seaborn', 'matplotlib'
        }
        
        name_lower = name.lower().replace('_', '-')
        
        if name_lower in core_packages or name in core_packages:
            return "core"
        elif name_lower in optional_packages or name in optional_packages:
            return "optional"
        else:
            return "dev"

    def _check_each_dependency(self):
        """Check each dependency's installation status"""
        for dep in self.dependencies:
            try:
                # Try to get installed version
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "show", dep.name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    # Package is installed
                    for line in result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            dep.installed_version = line.split(':', 1)[1].strip()
                            dep.status = DependencyStatus.INSTALLED
                            break
                else:
                    dep.status = DependencyStatus.MISSING
                    
            except Exception as e:
                dep.status = DependencyStatus.MISSING

    def get_summary(self) -> str:
        """Generate human-readable summary"""
        lines = []
        lines.append("=" * 70)
        lines.append("DEPENDENCY CHECK SUMMARY")
        lines.append("=" * 70)
        lines.append("")
        
        # Group by category
        core_deps = [d for d in self.dependencies if d.category == "core"]
        optional_deps = [d for d in self.dependencies if d.category == "optional"]
        dev_deps = [d for d in self.dependencies if d.category == "dev"]
        
        # Core dependencies
        lines.append("CORE DEPENDENCIES (required):")
        lines.append("-" * 70)
        
        core_installed = sum(1 for d in core_deps if d.status == DependencyStatus.INSTALLED)
        core_missing = sum(1 for d in core_deps if d.status == DependencyStatus.MISSING)
        
        for dep in sorted(core_deps, key=lambda x: x.name):
            icon = "OK" if dep.status == DependencyStatus.INSTALLED else "NOT FOUND"
            version = f"v{dep.installed_version}" if dep.installed_version else "NOT INSTALLED"
            lines.append(f"  {icon} {dep.name:25s} {version}")
        
        lines.append("")
        lines.append(f"Core: {core_installed}/{len(core_deps)} installed, {core_missing} missing")
        
        # Optional dependencies
        if optional_deps:
            lines.append("")
            lines.append("OPTIONAL DEPENDENCIES (enhance functionality):")
            lines.append("-" * 70)
            
            opt_installed = sum(1 for d in optional_deps if d.status == DependencyStatus.INSTALLED)
            
            for dep in sorted(optional_deps, key=lambda x: x.name):
                icon = "OK" if dep.status == DependencyStatus.INSTALLED else "MISSING"
                version = f"v{dep.installed_version}" if dep.installed_version else "not installed"
                lines.append(f"  {icon} {dep.name:25s} {version}")
            
            lines.append("")
            lines.append(f"Optional: {opt_installed}/{len(optional_deps)} installed")
        
        # Installation commands for missing
        missing_core = [d for d in core_deps if d.status == DependencyStatus.MISSING]
        if missing_core:
            lines.append("")
            lines.append("TO INSTALL MISSING CORE PACKAGES:")
            lines.append("-" * 70)
            lines.append(f"pip install {' '.join(d.name for d in missing_core)}")
        
        missing_optional = [d for d in optional_deps if d.status == DependencyStatus.MISSING]
        if missing_optional:
            lines.append("")
            lines.append("TO INSTALL OPTIONAL PACKAGES:")
            lines.append("-" * 70)
            lines.append(f"pip install {' '.join(d.name for d in missing_optional)}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)

    def install_missing(self, category: str = "core") -> bool:
        """
        Install missing dependencies.
        
        Args:
            category: "core", "optional", or "all"
            
        Returns:
            True if installation successful
        """
        missing = []
        
        if category == "all":
            missing = [d for d in self.dependencies if d.status == DependencyStatus.MISSING]
        else:
            missing = [d for d in self.dependencies 
                      if d.category == category and d.status == DependencyStatus.MISSING]
        
        if not missing:
            print(f"No missing {category} dependencies")
            return True
        
        print(f"\nInstalling {len(missing)} {category} packages...")
        packages = [d.name for d in missing]
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + packages,
                check=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            print(f"OK: Successfully installed {len(packages)} packages")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"FAILED: Installation failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            print(f"FAILED: Installation timed out")
            return False


def check_python_version() -> bool:
    """Check if Python version is compatible"""
    version = sys.version_info
    
    print("=" * 70)
    print("PYTHON VERSION CHECK")
    print("=" * 70)
    print(f"\nCurrent version: {version.major}.{version.minor}.{version.micro}")
    print(f"Required: Python 3.8 or higher\n")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("FAILED: Python version too old")
        print("  Please upgrade to Python 3.8 or higher")
        print("=" * 70)
        return False
    else:
        print("OK: Python version compatible")
        print("=" * 70)
        return True


def main():
    """Main entry point"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  DEPENDENCY VERIFICATION SYSTEM".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    # Check Python version
    if not check_python_version():
        return False
    
    print("\n")
    
    # Check dependencies
    checker = DependencyChecker()
    all_satisfied, dependencies = checker.check_all()
    
    print(checker.get_summary())
    
    # Offer to install missing
    if not all_satisfied:
        missing_core = [d for d in dependencies 
                       if d.category == "core" and d.status == DependencyStatus.MISSING]
        
        print(f"\nWARNING: {len(missing_core)} core dependencies missing")
        
        response = input("\nInstall missing core dependencies now? [Y/n]: ").strip().lower()
        
        if response in ['', 'y', 'yes']:
            success = checker.install_missing("core")
            if success:
                print("\nOK: Installation complete!")
                print("Run this script again to verify, or run: python scripts/validate.py")
            else:
                print("\nFAILED: Installation failed")
                print("Try manually: pip install -r requirements.txt")
    else:
        print("\nOK: All core dependencies satisfied!")
    
    print("\n")
    
    return all_satisfied


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
