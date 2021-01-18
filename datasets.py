#!/usr/bin/env python3

import json
import csv
import os
import logging

from collections import defaultdict, namedtuple
from utils import webnlg_parsing

RDFTriple = namedtuple('RDFTriple', 'subj pred obj')
logger = logging.getLogger(__name__)


class DataEntry:
    def __init__(self, triples, lexs):
        self.triples = triples
        self.lexs = lexs

    def __repr__(self):
        return str(self.__dict__)


class Dataset:
    def __init__(self, name):
        self.name = name
        self.data = {split: [] for split in ["train", "dev", "test"]}
        self.fallback_templates = ["The <predicate> of <subject> is <object> ."]

        # incremental examples will be extracted
        self.is_d2t = True

    def load_from_dir(self, path, splits):
        """Parses the original dataset files into an internal representation"""
        raise NotImplementedError

    def extract_templates(self, output_file):
        """Extract templates from the training data split"""
        raise NotImplementedError

    def load_templates(self, output_dir):
        """Loads existing templates from a JSON file"""
        templates_filename = os.path.join(output_dir, "templates.json")

        if os.path.isfile(templates_filename):
            logger.info(f"Loaded existing templates from {templates_filename}")
            with open(templates_filename) as f:
                self.templates_double = json.load(f)
        else:
            logger.info("Templates will be extracted from the training data")
            self.extract_templates(output_dir)

    def get_templates(self, triple):
        """Returns a set of templates for a given triple"""
        raise NotImplementedError

    def sort_by_lengths(self, datalist):
        datalist.sort(key=lambda entry: len(entry.triples))

        # find beginning of n-triples in the list for each n
        length_list = [len(entry.triples) for entry in datalist]
        lengths = sorted(list(set(length_list)))

        logger.info(f"Available lengths: {lengths}")

        idxs = [length_list.index(x) for x in lengths]

        logger.info(f"Starting indices: {idxs}")
        logger.info(f"Loaded {len(datalist)} items.")

        return lengths, idxs


class WebNLG(Dataset):
    def __init__(self):
        super().__init__(name="webnlg")

    def extract_templates(self, output_file):
        logger.info("Extracting templates")

        if not self.data["train"]:
            raise NotImplementedError(f"Templates can be extracted only from the train split.")

        l = []
        templates = defaultdict(set)
        data = self.data["train"]
        err = 0

        for entry in data:
            # templates are extracted only from 1-triple examples
            if len(entry.triples) != 1:
                continue

            triple = entry.triples[0]

            for lex in entry.lexs:
                ner2ent_items = list(lex["ner2ent"].items())

                if len(ner2ent_items) != 2:
                    err += 1
                    continue

                if ner2ent_items[0][1] == triple.subj and ner2ent_items[1][1] == triple.obj:
                    subj_ent = ner2ent_items[0][0]
                    obj_ent = ner2ent_items[1][0]
                elif ner2ent_items[0][1] == triple.obj and ner2ent_items[1][1] == triple.subj:
                    obj_ent = ner2ent_items[0][0]
                    subj_ent = ner2ent_items[1][0]
                else:
                    err += 1
                    continue

                template = lex["target"].replace(subj_ent, "<subject>") \
                                          .replace(obj_ent, "<object>")
                if template:
                    templates[triple.pred].add(template)

        if err > 0:
            logger.warning(f"Skipping {err} corrupted entries...")

        templates = {key: list(value) for key, value in templates.items()}

        with open(output_file, "w") as f:
            json.dump(templates, f,
                    indent=4, separators=(',', ': '), sort_keys=True)

        return templates


    def load_from_dir(self, data_dir, splits):
        for split in splits:
            logger.info(f"Loading {split} split")
            data_dir = os.path.join(path, split)
            err = 0
            xml_entryset = webnlg_parsing.run_parser(data_dir)

            for xml_entry in xml_entryset:
                triples = [RDFTriple(e.subject, e.predicate, e.object)
                    for e in xml_entry.modifiedtripleset]

                lexs = self._extract_lexs(xml_entry.lexEntries)

                if not any([lex['target_txt'] for lex in lexs]):
                    err += 1
                    continue

                entry = DataEntry(triples, lexs)
                self.data[split].append(entry)

            if err > 0:
                logger.warning(f"Skipping {err} entries without lexicalizations...")

    def get_templates(self, triple):
        pred = triple.pred

        if pred in self.templates_double:
            templates = self.templates_double[pred]
        else:
            templates = self.fallback_templates

        return templates


    def _extract_lexs(self, lex_entries):
        lexs = []

        for entry in lex_entries:
            target_txt = entry.text
            target = entry.template

            ner2ent = {
                ref.tag : ref.entity for ref in entry.references
            }
            lex = {
                "target_txt" : target_txt,
                "target" : target,
                "ner2ent" : ner2ent
            }
            lexs.append(lex)

        return lexs



class E2E(Dataset):
    def __init__(self):
        super().__init__(name="e2e")

    def load_from_dir(self, path, splits):
        for split in splits:
            logger.info(f"Loading {split} split")
            triples_to_lex = defaultdict(list)

            with open(os.path.join(path, f"{split}.csv")) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')

                # skip header
                next(csv_reader)
                err = 0

                for i, line in enumerate(csv_reader):
                    triples = self._mr_to_triples(line[0])

                    # probably a corrupted sample
                    if not triples or len(triples) == 1:
                        err += 1
                        # cannot skip for dev and test
                        if split == "train":
                            continue

                    lex = {
                        "target_txt" : line[1]
                    }
                    triples_to_lex[triples].append(lex)

                # triples are not sorted, complete entries can be created only after the dataset is processed
                for triples, lex_list in triples_to_lex.items():
                    entry = DataEntry(triples, lex_list)
                    self.data[split].append(entry)

            logger.warn(f"{err} corrupted instances")


    def load_templates(self, output_dir):
        """
        Loads existing templates from a JSON file
        Single-triple templates have to be created manually
        Double-triple templates can be extracted from the dataset (beware that data may be noisy)
        """
        templates_filename = os.path.join(output_dir, "templates.json")

        if os.path.isfile(templates_filename):
            logger.info(f"Loaded existing templates from {templates_filename}")
            with open(templates_filename) as f:
                self.templates = json.load(f)
        else:
            logger.error("Single-triple templates for the E2E dataset have to be created manually.")
            raise FileNotFoundError(templates_filename)

        templates_double_filename = os.path.join(output_dir, "templates_double.json")

        if os.path.isfile(templates_double_filename):
            logger.info(f"Loaded existing double templates from {templates_double_filename}")
            with open(templates_double_filename) as f:
                self.templates_double = json.load(f)
        else:
            self._extract_double_templates(templates_double_filename)


    def _mr_to_triples(self, mr):
        """
        Transforms E2E meaning representation into RDF triples.
        """
        triples = []

        # cannot be dictionary, slot keys can be duplicated
        items = [x.strip() for x in mr.split(",")]
        subj = None

        keys = []
        vals = []

        for item in items:
            key, val = item.split("[")
            val = val[:-1]

            keys.append(key)
            vals.append(val)

        name_idx = None if "name" not in keys else keys.index("name")
        eatType_idx = None if "eatType" not in keys else keys.index("eatType")

        # primary option: use `name` as a subject
        if name_idx is not None:
            subj = vals[name_idx]
            del keys[name_idx]
            del vals[name_idx]

            # corrupted case hotfix
            if not keys:
                keys.append("eatType")
                vals.append("restaurant")

        # in some cases, that does not work -> use `eatType` as a subject
        elif eatType_idx is not None:
            subj = vals[eatType_idx]
            del keys[eatType_idx]
            del vals[eatType_idx]
        # still in some cases, there is not even an eatType -> hotfix so that we do not lose data
        else:
            logger.warning(f"Cannot recognize subject in mr: {mr}")
            subj = "restaurant"

        for key, val in zip(keys, vals):
            triples.append(RDFTriple(subj, key, val))

        # will be used as a key in a dictionary
        return tuple(triples)


    # def select_triples_for_double_template(self, triples):
    #     sorted_triples = sorted(triples, key=lambda t:t[1])
    #     triple_pairs = [(s,t,i,j) for j,t in enumerate(sorted_triples) for i,s in enumerate(sorted_triples) if s[1] < t[1]]

    #     templates = list(filter(lambda x: x[0] is not None,
    #         [(self.get_templates_multiple(tp[:2]),tp[2],tp[3]) for tp in triple_pairs]))

    #     if templates:
    #         return random.choice(templates)

    #     return None


    def get_key_multiple(self, sorted_triples):
        slots = []
        for triple in sorted_triples:
            # special case, boolean value
            if triple.pred == "familyFriendly":
                slot = ("is" if triple.obj == "yes" else "not") + "FamilyFriendly"
            else:
                slot = triple.pred

            slots.append(slot)

        key = ", ".join(slots)
        return key


    def get_templates_multiple(self, triples):
        sorted_triples = sorted(triples, key=lambda t:t[1])

        key = self.get_key_multiple(sorted_triples)

        if key in self.templates_double:
            templates = self.templates_double[key]

            # TODO find a better way for speeding it up
            max_templates = 50
            if len(templates) > max_templates:
                templates = random.sample(templates, max_templates)

            return templates

        return None


    def get_templates(self, triple):
        if triple.pred in self.templates:
            templates = self.templates[triple.pred]

            if type(templates) is dict and triple.obj in templates:
                templates = templates[triple.obj]
        else:
            templates = self.fallback_templates
        return templates


    def _extract_double_templates(self, output_file):
        """
        Extract all templates for pairs of predicates
        """
        logger.info("Extracting double templates for the E2E dataset.")

        if not self.data["train"]:
            raise NotImplementedError(f"Double templates can be extracted only from the train split.")

        l = []
        entry_list = self.data["train"]
        templates = defaultdict(set)

        for entry in entry_list:
            n = len(entry.triples)

            # extract only pairs
            if n != 2:
                continue

            sorted_triples = sorted([triple for triple in entry.triples], key=lambda t: t.pred)

            # special case, bool value
            for triple in sorted_triples:
                if triple.pred == "familyFriendly":
                    pred = ("is" if triple.obj == "yes" else "not") + "FamilyFriendly"
                    # should not appear in the sentence
                    obj = "<plh>"
                    triple = RDFTriple(triple.subj, pred, obj)

            key = self.get_key_multiple(sorted_triples)
            triple1, triple2 = sorted_triples

            assert triple1.subj == triple2.subj

            lexicalizations = [lex['target_txt'] for lex in entry.lexs]

            for lex in lexicalizations:
                if triple1.subj in lex and \
                    (triple1.obj in lex or triple1.obj == "<plh>") and \
                    (triple2.obj in lex or triple2.obj == "<plh>"):
                    template = lex.replace(triple1.subj, "<subject>") \
                                          .replace(triple1.obj, "<object1>") \
                                          .replace(triple2.obj, "<object2>")
                    templates[key].add(template)

        templates = {key: list(value) for key, value in templates.items()}

        with open(output_file, "w") as json_file:
            json.dump(templates,json_file,
                    indent=4, separators=(',', ': '), sort_keys=True)

        return templates



class DiscoFuse(Dataset):
    def __init__(self):
        super().__init__(name="discofuse")
        self.is_d2t = False

    def load_from_dir(self, path, splits):
        for split in splits:
            logger.info(f"Loading {split} split")

            with open(f"{path}/{split}_balanced.tsv", "r") as f_in:
                # skip header
                next(f_in)

                selected = 0
                skipped = 0

                for line in f_in:
                    cols = line.strip().split("\t")

                    disco_type = cols[4]
                    connective = cols[5]

                    if disco_type in ["PAIR_ANAPHORA", "SINGLE_APPOSITION", "SINGLE_RELATIVE", "PAIR_NONE"] \
                        or (disco_type in ["SINGLE_S_COORD", "SINGLE_S_COORD_ANAPHORA", "SINGLE_VP_COORD"]
                            and connective in ["and", ", and"]):
                        src = " ".join(cols[2:4])
                        tgt = " ".join(cols[0:2])

                        self.data[split].append((src, tgt))
                        selected += 1
                    else:
                        skipped += 1

                logger.info(f"{split} processed: {selected} selected, {skipped} skipped")

    def extract_templates(self, output_file):
        """
        Templates are not used with the DiscoFuse dataset
        """
        pass

