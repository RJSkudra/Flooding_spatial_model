# Rendering configuration settings
import os
import sys
import json

# Default configuration
DEFAULT_CONFIG = {
    "backend": "QtAgg",  # Default backend (QtAgg, TkAgg, or Agg)
    "save_dir": "rendered_output",  # Directory to save rendered images
    "auto_save": True,  # Always save rendered images to disk
    "auto_display": True,  # Try to display images interactively
    "figure_dpi": 150  # Resolution for saved images
}

CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "rendering_config.json"
)

def load_config():
    """Load rendering configuration from file or create with defaults"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

def save_config(config):
    """Save rendering configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving configuration: {e}")

def setup_rendering(matplotlib=None):
    """Configure matplotlib rendering based on settings"""
    config = load_config()
    
    # Create output directory if it doesn't exist
    if config["auto_save"]:
        save_dir = config["save_dir"]
        # Handle both relative and absolute paths
        if not os.path.isabs(save_dir):
            # Make path relative to the project root, not the current directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir = os.path.join(project_root, save_dir)
        
        os.makedirs(save_dir, exist_ok=True)
    
    # Configure matplotlib if provided
    if matplotlib:
        try:
            # Try using the configured backend
            matplotlib.use(config["backend"])
        except Exception as backend_error:
            print(f"{config['backend']} backend failed: {backend_error}")
            matplotlib.use('Agg')
            config["auto_display"] = False
            save_config(config)
    
    return config

def get_output_path(filename):
    """Get the full path for an output file"""
    config = load_config()
    save_dir = config["save_dir"]
    
    # Handle both relative and absolute paths
    if not os.path.isabs(save_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, save_dir)
    
    return os.path.join(save_dir, filename)

if __name__ == "__main__":
    # If run directly, allow updating configuration
    import argparse
    
    parser = argparse.ArgumentParser(description="Configure rendering settings")
    parser.add_argument("--backend", choices=["QtAgg", "TkAgg", "Agg"], 
                        help="Matplotlib backend to use")
    parser.add_argument("--save-dir", help="Directory to save rendered images")
    parser.add_argument("--auto-save", choices=["True", "False"], 
                        help="Always save rendered images")
    parser.add_argument("--auto-display", choices=["True", "False"], 
                        help="Try to display images interactively")
    parser.add_argument("--figure-dpi", type=int, help="Resolution for saved images")
    
    args = parser.parse_args()
    
    # Load current config
    config = load_config()
    
    # Update config with any provided arguments
    if args.backend:
        config["backend"] = args.backend
    if args.save_dir:
        config["save_dir"] = args.save_dir
    if args.auto_save:
        config["auto_save"] = args.auto_save == "True"
    if args.auto_display:
        config["auto_display"] = args.auto_display == "True"
    if args.figure_dpi:
        config["figure_dpi"] = args.figure_dpi
    
    # Save updated config
    save_config(config)