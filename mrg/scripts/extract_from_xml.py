import os
import json
import argparse
import xml.etree.ElementTree as ET

def parse_xml_to_json(folder, top_n=5, verbose=True):
    """Based on:
    https://github.com/nlpaueb/bio_image_caption/blob/master/SiVL19/get_iu_xray.py
    """
    # Reports to read
    reports_names = os.listdir(folder)
    if top_n is not None and top_n > 0:
        reports_names = reports_names[:top_n]

    # To save info
    parsed_reports = dict()

    # Read report sections
    read_sections = ["comparison", "indication", "findings", "impression"]

    # To report errors
    reports_with_no_image = []
    reports_with_no_tags = []
    reports_with_empty = { key: [] for key in read_sections }

    for report_name in reports_names:
        report_fname = os.path.join(folder, report_name)
        tree = ET.parse(report_fname)
        root = tree.getroot()

        report_dict = {
            "filename": report_name,
        }

        # find the images of the report
        images_xml = root.findall("parentImage")
        images = []
        for image_xml in images_xml:
            image_id = image_xml.get("id")
            image_caption = image_xml.find("caption").text
            images.append({
                "id": image_id,
                "caption": image_caption,
            })
        report_dict["images"] = images

        if len(images) == 0:
            reports_with_no_image.append(report_fname)

        # find impression and findings sections
        sections = root.find("MedlineCitation").find("Article").find("Abstract").findall("AbstractText")
        for section in sections:
            label = section.get("Label").lower()
            if label in read_sections:
                text = section.text
                report_dict[label] = text

        for section in read_sections:
            if section not in report_dict:
                reports_with_empty[section].append(report_fname)

        # get the MESH tags
        tags = root.find("MeSH")
        if tags is not None:
            report_dict["tags_manual"] = [t.text for t in tags.findall("major")]
            report_dict["tags_auto"] = [t.text for t in tags.findall("automatic")]
        else:
            reports_with_no_tags.append(report_fname)

        parsed_reports[report_name] = report_dict

    def print_errors(l, name):
        print(f"{name}: {len(l)}")
        for r in l:
            print(f"\t{r}")

    if verbose:
        print_errors(reports_with_no_image, "no image")
        print_errors(reports_with_no_tags, "no tags")
        for section in read_sections:
            print_errors(reports_with_empty[section], f"no {section}")

    return parsed_reports

def save_json(obj, filename, pretty=True):
    kwargs = {}
    if pretty:
        kwargs["indent"] = 2
        kwargs["sort_keys"] = True
    with open(filename, "w") as f:
        json.dump(obj, f, **kwargs)
    print(f"JSON saved to: {filename}")

def parse_args():
    base_dir = os.path.join(
        os.environ.get("DATASET_DIR_IU_XRAY", ""),
        "reports",
    )

    parser = argparse.ArgumentParser(
        description="Parses IU-x-ray reports to JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("-b", "--base-dir", type=str, default=base_dir,
                        help="Base directory where the dataset is located. Defaults to env variable '${DATASET_DIR_IU_XRAY}/reports'")
    parser.add_argument("-f", "--folder", type=str, default="ecgen-radiology",
                        help="Folder to read the reports from")
    parser.add_argument("-n", type=int, default=None, help="Amount of files to read (for debugging)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Whether to print errors details")
    parser.add_argument("-o", "--output", type=str, default="reports.json",
                        help="Filepath to write the reports json")
    parser.add_argument("-m", "--min", action="store_true", help="whether to minify the file")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    parsed_reports = parse_xml_to_json(args.folder, top_n=args.n, verbose=args.verbose)

    output_filepath = os.path.join(args.base_dir, args.output)
    save_json(parsed_reports, output_filepath, pretty=not args.min)
