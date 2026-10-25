import os
from scandir_rs import Walk

def rust_rglob_walk(root_dir, target_extension):
    if not target_extension.startswith("."):
        target_extension = f".{target_extension}"
        
    # Standard string path iteration
    walker = Walk(root_dir, store=False)
    matches = []
    
    for root, dirs, files in walker:
        for file in files:
            if file.lower().endswith(target_extension.lower()):
                matches.append(os.path.join(root, file))
                
    return matches


# --- Example Usage ---
if __name__ == "__main__":
    search_path = "vllm-0.10.1/docs" 
    
    found_files = rust_rglob_walk(search_path, "md")
    
    print(f"⚡ Found {len(found_files)} files.")
    
    for path in found_files[:5]:
        print(f" -> {path}")
