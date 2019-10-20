# -*- coding: utf-8 -*-
"""Summary

Attributes:
    body (TYPE): Description
    constructed_url (TYPE): Description
    data (TYPE): Description
    endpoint (TYPE): Description
    endpoint_var_name (str): Description
    headers (TYPE): Description
    key_var_name (str): Description
    params (TYPE): Description
    path (TYPE): Description
    request (TYPE): Description
    response (TYPE): Description
    subscription_key (TYPE): Description
    val (TYPE): Description
"""

# This simple app uses the '/translate' resource to translate text from
# one language to another.

# This sample runs on Python 2.7.x and Python 3.x.
# You may need to install requests and uuid.
# Run: pip install requests uuid

import ast
import re
import os
import csv
import requests
import uuid
import json

from .. import util

key_var_name = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY'
if not (key_var_name in os.environ):
    raise Exception(
        'Please set/export the environment variable: {}'.format(key_var_name))
subscription_key = os.environ[key_var_name]

endpoint_var_name = 'TRANSLATOR_TEXT_ENDPOINT'
if not (endpoint_var_name in os.environ):
    raise Exception(
        'Please set/export the environment variable: {}'
        .format(endpoint_var_name))
endpoint = os.environ[endpoint_var_name]

# If you encounter any issues with the base_url or path, make sure
# that you are using the latest endpoint:
# https://docs.microsoft.com/azure/cognitive-services/translator/reference/v3-0-translate
path = '/translate?api-version=3.0'

headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

CONST_DELIMITER = '。'
CONST_PARENTHESES = ['「', '」'], ['『', '』']

def translate_azure(
        input_file, output_file, start_index=1, end_index=-1,
        dst='ja', display_every=0,
        max_retries=1):
    """
    Function to batch-translate sentences from %input_file% and save outputs
        to file %output_file%

    Args:
        input_file (str): Filepath of file containing sentences to translate
        output_file (TYPE): Filepath to file where translations will be written
        dst (str, optional): Desired output language
        display_every (int, optional): Frequency of outputs
        max_retries (int, optional): Maximum number of retries to API
    """
    print('Reading text from file: %s' % input_file)
    params = '&to=%s' % dst
    constructed_url = endpoint + path + params

    util.mkdir_p(output_file, file=True)

    f_input = open(input_file, 'r+')
    f_output = open(output_file, 'a+')

    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    last_notified_at = 0
    n_completed = 1

    for row in reader:

        if n_completed < start_index:
            n_completed += 1
            continue

        elif end_index != -1 and n_completed > end_index:
            break

        passed = False
        retries = 0

        while not passed and retries < max_retries:

            try:

                # You can pass more than one object in body.
                body = [{
                    'text': '%s' % row
                }]

                request = requests.post(
                    constructed_url, headers=headers, json=body)

                response = request.json()
                data = json.dumps(response, sort_keys=True,
                                  indent=4, separators=(',', ': '))
                data = json.loads(data)
                writer.writerow(data)

                retries += 1
                passed = True

            except Exception:

                print('WARNING: API call failed. Retrying x%1d' % retries)
                continue

        if passed:

            n_completed += 1

            if n_completed - last_notified_at > display_every:

                print('Sentences completed: %2d' % n_completed)
                last_notified_at = n_completed

        else:

            print('WARNING: Sentence at index %2d \
                   failed with %2d retries' % (n_completed, max_retries))

    f_input.close()
    f_output.close()


def convert_azure_data(input_file, output_file):

    input_file = open(input_file, 'r')
    output_file = open(output_file, 'w+')

    reader = csv.reader(input_file)

    template = re.compile(r"(.+)(text':)(.+)(, 'to':)(.+)")

    for row in reader:

        row = row[0]
        row = template.sub('\\3', row)
        row = re.sub(r'[[\.\,\'\"\[\]\(\)\?\!\\\s+]', '', row)

        for parentheses in CONST_PARENTHESES:

            l_p = parentheses[0]
            r_p = parentheses[1]

            # Fix mismatched parentheses
            left = row.count(l_p)
            right = row.count(r_p)

            if left == right:
                if row[0] == l_p and row[-1] == r_p:
                    row = row[1:-1]

                continue

            elif left == right + 1 and row[0] == l_p:
                row = row[1:]

            elif right == left + 1 and row[-1] == r_p:
                row = row[:-1]

            else:
                row = row.replace(r_p, '')
                row = row.replace(l_p, '')

        if row[-1] != CONST_DELIMITER:
            row += CONST_DELIMITER

        output_file.write(row + os.linesep)

    input_file.close()
    output_file.close()


def translate_azure_tanaka():
    """Summary
    """
    input_file = './raw_data/tanaka/en.csv'
    output_file = './raw_data/translated/tanaka_azure.csv'

    translate_azure(input_file, output_file, start_index=148934, end_index=-1)
