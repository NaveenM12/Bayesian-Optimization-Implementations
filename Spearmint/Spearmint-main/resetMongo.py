#!/usr/bin/python

"""
Quick and easy fix for hanging issues at Spearmint
(see here https://github.com/HIPS/Spearmint/issues/25)
"""
import json
import logging

from pymongo import MongoClient

logging.basicConfig()
logger = logging.getLogger(__name__)


try:
    logger.info('Using config.json')
    with file('/Users/naveenmirapuri/PycharmProjects/SpearmintImplementation/Spearmint-master/MNIST/config.json') as fin:
        config = json.load(fin)

except IOError:
    logger.error('config.json was not found on the path.\
Are you sure this is the correct directory?')
    exit(1)


experiment_name = config['experiment-name']
table = '{}.jobs'.format(experiment_name)
logger.info('Using experiment: %s' % experiment_name)

try:
    logger.info('Establishing connection to localhost')
    client = MongoClient()
except:
    logger.error('Could not establish connection')
    exit(1)

try:
    logger.info('Establishing connection to spearmint')
    db = client['spearmint']
except:
    logger.error('Spearmint collection not found...you sure \
it started properly?')

num_jobs = len([x for x in db[table].find({'status': 'pending'})])
logger.info('Found %d jobs to delete...' % num_jobs)
db[table].delete_many({'status': 'pending'})
