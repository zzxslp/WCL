"""Entry-point script to label radiology reports."""
import pandas as pd

from args import ArgParser
from loader import LoaderOnFly
from stages import Extractor, Classifier, Aggregator
from constants import *
import json
import os
from tqdm import tqdm

IMAGE_IDS = 'image_ids'

def write(reports, labels, image_ids, output_path, verbose=False):
    """Write labeled reports to specified path."""
    labeled_reports = pd.DataFrame({REPORTS: reports, IMAGE_IDS: image_ids})
    for index, category in enumerate(CATEGORIES):
        labeled_reports[category] = labels[:, index]

    if verbose:
        print(f"Writing reports and labels to {output_path}.")
    labeled_reports[[IMAGE_IDS, REPORTS] + CATEGORIES].to_csv(output_path,
                                                   index=False)

def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def label(args):
    print(args)
    """Label the provided report(s)."""

    loader = LoaderOnFly(args.extract_impression)

    extractor = Extractor(args.mention_phrases_dir,
                          args.unmention_phrases_dir,
                          verbose=args.verbose)
    classifier = Classifier(args.pre_negation_uncertainty_path,
                            args.negation_path,
                            args.post_negation_uncertainty_path,
                            verbose=args.verbose)
    aggregator = Aggregator(CATEGORIES,
                            verbose=args.verbose)

    # read reports from other sources
    # reports = ['no acute disease']
    reports = []
    image_ids = []

    file_name = str(args.reports_path).split('/')[-1]
    if file_name == 'full_captions.json': 
        with open(args.reports_path, 'r') as f:
            data = json.load(f)
            for k, d in data.items():
                sents = d['sents'][:-1] # skip indication
                sents = [s + '.' for s in sents]
                reports.append(' '.join(sents))
                image_ids.append(k)
                
    elif file_name == 'full_data.json':
        with open(args.reports_path, 'r') as f:
            for l in f:
                d = json.loads(l)
                if 'image_id' in d and len(d['image_id']) != 0:
                    temp = d['image_id']
                    image_id = temp[0].split('_')[0]

                    impression = d.get('IMPRESSION', '') or ''
                    findings = d.get('FINDINGS', '') or ''

                    try:
                        if impression[-1] != '.':
                            impression = impression + '.'
                        if findings[-1] != '.':
                            findings = findings + '.'
                    except:
                        pass
                    reports.append(impression + ' ' + findings)
                    image_ids.append(image_id)

    elif 'debug' in file_name or 'sample' in file_name:
        test = load_json(args.reports_path)
        for i, image_id in enumerate(test):
            array = []
            if args.real_sent:
                for each in test[image_id]['Real Sent']:
                    sent = test[image_id]['Real Sent'][each]
                    if sent:
                        array.append(sent)
            else:
                for each in test[image_id]['Pred Sent']:
                    sent = test[image_id]['Pred Sent'][each]
                    if sent:
                        array.append(sent)
            pred_sent = '. '.join(array)
            reports.append(pred_sent)
            image_ids.append(image_id)

    else: ## my generated reports format {image_id: report}
        pred_captions = load_json(args.reports_path)
        for i, image_id in enumerate(pred_captions):
            pred_sent = pred_captions[image_id]
            reports.append(pred_sent)
            image_ids.append(image_id)

    
    print(reports[:3])

    # Load reports in place.
    loader.load(reports)
    # Extract observation mentions in place.
    extractor.extract(loader.collection)
    # Classify mentions in place.
    classifier.classify(loader.collection)
    # Aggregate mentions to obtain one set of labels for each report.
    labels = aggregator.aggregate(loader.collection)

    write(loader.reports, labels, image_ids, args.output_path, args.verbose)


if __name__ == "__main__":
    parser = ArgParser()
    label(parser.parse_args())
