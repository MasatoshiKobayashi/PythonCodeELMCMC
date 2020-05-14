'''
Uploads electron lifetime trend to run database

'''

import os
import glob
import socket
from pymongo import MongoClient
import datetime
import argparse


def getMostRecentFile(pathToFiles):
    '''

    Pulls more recent file from directory

    Args:
        pathToFiles (str): path to directory where electron lifetime trends
                            are stored

    '''
    
    FileList = pathToFiles + '*'
    SortedFiles = sorted(glob.iglob(FileList), key=os.path.getctime, reverse=True)

    try:
        MostRecentFile = SortedFiles[0]
        print('\n===>  Loaded most recent file: %s\n' %MostRecentFile)
        return MostRecentFile
    except:
        print('\nWarning: no prediction file!')

    return ''


def uploadTrendToDatabase(user, pathToERTrend, pathToAlphaTrend, description='', **kwargs):
    '''

    Uploads trend to corrections database

    Args:
        pathToERTrend (str): path to electronic recoil trend

        pathToAlphaTrend (str): path to alpha trend

        description (str): description of trend

    kwargs:
        major_version_bump (bool): new trend upgrade in trend.  Automatically
                                    resets minor version to 0.  Default: False

        minor_version_bump (bool): new trend upgrade in trend.  Default: True

    '''

    major_version_bump = kwargs.get('major_version_bump', False)
    minor_version_bump = kwargs.get('minor_version_bump', True)

    if 'password' not in kwargs:
        try:
            password = os.environ['CORRDBPASS']
        except:
            raise KeyError('Must give \'password\' in kwargs or set environment variable \'CORRDBPASS\'')
    else:
        password = kwargs.get('password')

    database = 'run'
    collection = 'hax_electron_lifetime'


    client = MongoClient("mongodb://corrections:%s@xenon1t-daq.lngs.infn.it:27017/run"%password)
    c = client[database][collection]

    cur = c.find().sort("creation_time", -1)
    major = 0
    minor = 0
    if cur.count() != 0:
        d = cur[0]
        major = int(d['version'].split('.')[0])
        minor = int(d['version'].split('.')[1])
    if major_version_bump:
        major += 1
        minor = 0
    elif minor_version_bump:
        minor += 1
    version = ('%i.%i'%(major, minor))

    erTrend = open(pathToERTrend).readlines()
    alphaTrend = open(pathToAlphaTrend).readlines()

    print('\n\n===========> Using ' + '/'.join(pathToERTrend.split('/')[5:]) + '!\n')
    print('\n===========> Using ' + '/'.join(pathToAlphaTrend.split('/')[5:]) + '!\n')

    doc = {
        'name': 'electron_lifetime_correction',
        'user': user,
        'creation_time': datetime.datetime.utcnow(),
        'version': version,
        'description': description,
        'times': [],
        'electron_lifetimes': [],
        'electron_lifetime_errors_low': [],
        'electron_lifetime_errors_up': [],
        'times_alpha': [],
        'electron_lifetimes_alpha': [],
        'electron_lifetime_error_alphas_low': [],
        'electron_lifetime_error_alphas_up': []
    }

    for line in erTrend:
        values = line[:-1].split('\t\t')
        doc['times'].append(float(values[0]))
        doc['electron_lifetimes'].append(float(values[1]))
        doc['electron_lifetime_errors_low'].append(float(values[2]))
        doc['electron_lifetime_errors_up'].append(float(values[3]))
    
    for line in alphaTrend:
        values = line[:-1].split('\t\t')
        doc['times_alpha'].append(float(values[0]))
        doc['electron_lifetimes_alpha'].append(float(values[1]))
        doc['electron_lifetime_error_alphas_low'].append(float(values[2]))
        doc['electron_lifetime_error_alphas_up'].append(float(values[3]))
    
    print(doc['version'])

    response = input('Are you sure you want to upload to the runs database? (y/n)')

    if response.lower() == 'y':
        c.insert(doc)
        print('Uploaded')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('user', type=str, help='User')
    parser.add_argument('--password', type=str, help='Password for runs database (default is environment variable \'CORRDBPASS\')')
    parser.add_argument('--er', type=str, help='ER lifetimes')
    parser.add_argument('--alpha', type=str, help='Alpha lifetimes')
    parser.add_argument('--file_path', type=str, help='Path to lifetimes')
    parser.add_argument('--description', type=str, help='Description or comments')
    parser.add_argument('--major_version_bump', help='Increase major version (minor version reset to 0)', default=False)
    parser.add_argument('--minor_version_bump', help='Increase minor version', default=True)
    args = parser.parse_args()
    arg_dict = args.__dict__

    if args.file_path is None:
        arg_dict['file_path'] = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/MCMC_Results/TXTs/CorrectionsMinitrees/'

    if args.er is None:
        arg_dict['er'] = getMostRecentFile(arg_dict['file_path'] + 'Kr')
    if args.alpha is None:
        arg_dict['alpha'] = getMostRecentFile(arg_dict['file_path'] + 'Prediction')

    user = arg_dict.pop('user')
    er = arg_dict.pop('er')
    alpha = arg_dict.pop('alpha')
    description = arg_dict.pop('description')

    uploadTrendToDatabase(user, er, alpha, description=description, **arg_dict)
