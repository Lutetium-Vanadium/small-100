import argparse, sys

lang_list = ['af', 'am', 'ar', 'ast', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'ilo', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'lb', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'ns', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'th', 'tl', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zu']

parser = argparse.ArgumentParser(
                    prog='small-100',
                    description='translate text using SMaLL-100',
                    epilog='to run the program, pass the target language option then write to stdin')

parser.add_argument('-l', '--langs', action=argparse.BooleanOptionalAction, help="output available languages")
parser.add_argument('-t', '--target', help="target language")
values = parser.parse_args()

if values.langs:
    print(*lang_list)
    exit(0)

tgt_lang = values.target

if tgt_lang not in lang_list:
    print("Unknown target language specified:", tgt_lang)
    print("Run with --langs to see available langs")

from transformers import M2M100ForConditionalGeneration
from tokenization_small100 import SMALL100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")

def small100_tr(lang, text):
    tokenizer.tgt_lang = lang
    encoded_text = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text)
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

for line in sys.stdin:
    print(small100_tr(tgt_lang, line.strip()))
