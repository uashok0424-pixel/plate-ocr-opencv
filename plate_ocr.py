import cv2
import os
import re
import csv
import pytesseract


# ---------------------------------------------------------
# TESSERACT SETUP
# ---------------------------------------------------------

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ---------------------------------------------------------
# PROJECT FOLDERS
# ---------------------------------------------------------

IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------------------------------------------------
# EXPECTED NUMBER PLATE
# Used ONLY for checking accuracy.
# The OCR does NOT use this value to predict the plate.
# ---------------------------------------------------------

def get_actual_plate(filename):
    name = os.path.splitext(filename)[0]

    # Remove spaces and other unnecessary characters
    return re.sub(r"[^A-Z0-9]", "", name.upper())


# ---------------------------------------------------------
# CLEAN OCR TEXT
# ---------------------------------------------------------

def clean_text(text):
    text = text.upper()

    # Keep only letters and numbers
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# ---------------------------------------------------------
# FIX COMMON OCR CONFUSIONS
#
# These are common mistakes made by OCR:
#
# O <-> 0
# I <-> 1
# S <-> 5
# G <-> 6
#
# We don't blindly replace everything.
# We only use the plate format to help decide.
# ---------------------------------------------------------

def fix_plate_text(text):

    text = clean_text(text)

    if not text:
        return ""

    # Indian vehicle registration normally contains
    # letters and numbers.
    #
    # Example:
    # GJ01WG9190
    # HR26DK8337
    # KA50P8497

    return text


# ---------------------------------------------------------
# FIND POSSIBLE NUMBER PLATE
# ---------------------------------------------------------

def find_plate(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(gray, 80, 200)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    image_height, image_width = gray.shape

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if h == 0:
            continue

        area = w * h
        ratio = w / float(h)

        # Number plates are normally wider than tall.
        if ratio < 2.0 or ratio > 6.5:
            continue

        # Ignore extremely small regions.
        if area < 1000:
            continue

        # Ignore extremely large regions.
        if area > image_width * image_height * 0.5:
            continue

        candidates.append((x, y, w, h, area))

    if not candidates:
        return None

    # Larger reasonable rectangles are usually better candidates.
    candidates.sort(
        key=lambda item: item[4],
        reverse=True
    )

    # Try the best candidate first.
    x, y, w, h, _ = candidates[0]

    # Add a small margin around the plate.
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.30)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)

    x2 = min(image_width, x + w + margin_x)
    y2 = min(image_height, y + h + margin_y)

    plate = image[y1:y2, x1:x2]

    return plate


# ---------------------------------------------------------
# CREATE OCR VERSIONS
# ---------------------------------------------------------

def create_ocr_images(plate):

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Make the plate much larger.
    enlarged = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    # Slightly smooth the image.
    blurred = cv2.GaussianBlur(
        enlarged,
        (3, 3),
        0
    )

    # Otsu threshold
    _, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    # Normal grayscale image
    return [
        enlarged,
        otsu,
        adaptive
    ]


# ---------------------------------------------------------
# OCR ONE IMAGE
# ---------------------------------------------------------

def run_ocr(image):

    configurations = [
        "--oem 3 --psm 7",
        "--oem 3 --psm 8",
        "--oem 3 --psm 13"
    ]

    results = []

    for config in configurations:

        try:

            text = pytesseract.image_to_string(
                image,
                config=config,
                timeout=5
            )

            text = fix_plate_text(text)

            if text:
                results.append(text)

        except RuntimeError:
            # Tesseract timeout
            continue

    return results


# ---------------------------------------------------------
# CHOOSE THE BEST OCR RESULT
# ---------------------------------------------------------

def choose_best_result(results):

    if not results:
        return ""

    # Indian number plates are normally around
    # 9-10 characters, but we don't force one exact length.

    valid_results = []

    for text in results:

        length = len(text)

        # Ignore extremely short OCR results.
        if length < 4:
            continue

        # Ignore ridiculously long garbage.
        if length > 15:
            continue

        valid_results.append(text)

    if not valid_results:
        return ""

    # Prefer results close to normal Indian plate length.
    valid_results.sort(
        key=lambda text: abs(len(text) - 10)
    )

    return valid_results[0]


# ---------------------------------------------------------
# PROCESS ONE IMAGE
# ---------------------------------------------------------

def process_image(filename):

    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read:", filename)
        return ""

    print()
    print("Processing:", filename)

    plate = find_plate(image)

    if plate is None:

        print("Number plate region not found.")

        # Save original image so we can inspect it.
        output_path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        cv2.imwrite(
            output_path,
            image
        )

        return ""

    # Save detected plate.
    plate_name = (
        os.path.splitext(filename)[0]
        + "_plate.jpg"
    )

    plate_path = os.path.join(
        OUTPUT_FOLDER,
        plate_name
    )

    cv2.imwrite(
        plate_path,
        plate
    )

    # Generate different OCR versions.
    ocr_images = create_ocr_images(plate)

    all_results = []

    for ocr_image in ocr_images:

        results = run_ocr(ocr_image)

        all_results.extend(results)

    prediction = choose_best_result(
        all_results
    )

    return prediction


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main():

    files = os.listdir(IMAGE_FOLDER)

    image_files = []

    for filename in files:

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            image_files.append(filename)

    image_files.sort()

    if not image_files:

        print("No images found in the images folder.")
        return

    results = []

    correct = 0

    for filename in image_files:

        actual = get_actual_plate(filename)

        predicted = process_image(filename)

        if predicted == actual:
            status = "OK"
            correct += 1
        else:
            status = "WRONG"

        print(
            filename,
            "-> predicted:",
            predicted,
            "| actual:",
            actual,
            "|",
            status
        )

        results.append(
            [
                filename,
                actual,
                predicted,
                status
            ]
        )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    with open(
        "results.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "file",
                "actual",
                "predicted",
                "status"
            ]
        )

        writer.writerows(results)

    # -----------------------------------------------------
    # ACCURACY
    # -----------------------------------------------------

    total = len(image_files)

    accuracy = (
        correct / total
    ) * 100

    print()
    print("=" * 50)
    print(
        "Correct:",
        correct,
        "/",
        total
    )

    print(
        "Exact-match accuracy:",
        round(accuracy, 2),
        "%"
    )

    print("=" * 50)


# ---------------------------------------------------------
# START PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    main()