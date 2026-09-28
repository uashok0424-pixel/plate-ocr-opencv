import cv2
import os
import re
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

OUTPUT = "output"

PLATES = [
    "GJ01WG9190",
    "HR26DK8337",
    "KA50P8497",
    "RJ27GD4827",
    "UP80DJ277",
    "WB06F5977"
]


def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def run_ocr(img, psm):
    config = (
        f"--psm {psm} "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    try:
        text = pytesseract.image_to_string(
            img,
            config=config,
            timeout=5
        )
        return clean_text(text)

    except Exception as e:
        return "ERROR"


def create_variants(image):
    variants = []

    # --------------------------------------------------
    # 1. GRAYSCALE
    # --------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------
    # 2. UPSCALE
    # --------------------------------------------------
    gray = cv2.resize(
        gray,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    variants.append(("gray", gray))

    # --------------------------------------------------
    # 3. NORMAL THRESHOLD
    # --------------------------------------------------
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(("otsu", binary))

    # --------------------------------------------------
    # 4. INVERTED THRESHOLD
    # --------------------------------------------------
    inverted = cv2.bitwise_not(binary)

    variants.append(("inverted", inverted))

    # --------------------------------------------------
    # 5. ADAPTIVE THRESHOLD
    # --------------------------------------------------
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    variants.append(("adaptive", adaptive))

    # --------------------------------------------------
    # 6. CLAHE
    # --------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    variants.append(("clahe", enhanced))

    # --------------------------------------------------
    # 7. SHARPEN
    # --------------------------------------------------
    blur = cv2.GaussianBlur(gray, (0, 0), 3)

    sharpened = cv2.addWeighted(
        gray,
        1.8,
        blur,
        -0.8,
        0
    )

    variants.append(("sharp", sharpened))

    return variants


def main():

    print("=" * 70)
    print("PLATE OCR MULTI-PREPROCESSING TEST")
    print("=" * 70)

    for plate_name in PLATES:

        filename = os.path.join(
            OUTPUT,
            plate_name + "_plate.jpg"
        )

        print()
        print("-" * 70)
        print("IMAGE:", plate_name)
        print("-" * 70)

        if not os.path.exists(filename):
            print("FILE NOT FOUND")
            continue

        image = cv2.imread(filename)

        if image is None:
            print("IMAGE COULD NOT BE READ")
            continue

        variants = create_variants(image)

        for variant_name, variant in variants:

            results = []

            for psm in [6, 7, 8, 13]:

                result = run_ocr(
                    variant,
                    psm
                )

                if result:
                    results.append(
                        f"PSM{psm}={result}"
                    )

            print(
                f"{variant_name:10} -> "
                + " | ".join(results)
            )

    print()
    print("=" * 70)
    print("TEST FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()