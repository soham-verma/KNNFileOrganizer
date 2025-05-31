import os
import shutil

# Define categories and matching file keywords
categories = {
    "Education": [
        "Diploma", "Transcript", "Testamur", "CoE", "ETRF", "IELTS"
    ],
    "Immigration & Visa": [
        "IMMI", "COMP LETTER", "Bridging Visa", "Grant", "AttachmentA", "14070443PC"
    ],
    "Identification": [
        "Passport", "photo_id", "driving_licence"
    ],
    "Health Insurance": [
        "Bupa", "Medibank"
    ],
    "Banking & Finance": [
        "commbank"
    ],
    "Personal": [
        "Address", "address.txt"
    ],
    "Projects": [
        "BaseStation", "Build"
    ]
}

# Get current directory
current_dir = os.getcwd()

# Helper function to move files
def move_file(file_name, folder_name):
    target_folder = os.path.join(current_dir, folder_name)
    os.makedirs(target_folder, exist_ok=True)
    shutil.move(os.path.join(current_dir, file_name), os.path.join(target_folder, file_name))

# Go through each file and move it
for file in os.listdir(current_dir):
    if os.path.isfile(os.path.join(current_dir, file)):
        moved = False
        for category, keywords in categories.items():
            if any(keyword in file for keyword in keywords):
                move_file(file, category)
                print(f"Moved '{file}' to '{category}/'")
                moved = True
                break
        if not moved:
            move_file(file, "Uncategorized")
            print(f"Moved '{file}' to 'Uncategorized/'")

print("\n✅ Files organized.")
