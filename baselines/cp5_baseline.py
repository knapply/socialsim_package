import pandas as pd
import json

import socialsim as ss

import glob
import re
import datetime
import numpy as np
import uuid
import pprint
import time
import random


def add_new_users(data, hist_data, previous_users = []):
    
    print('Adding new users...')
    
    #Get the min and max dates from historical data, and compute new user rates based on 24 hr subsets from min to max dates
    new_users = hist_data[~hist_data['nodeUserID'].isin(previous_users)]
    hist_min,hist_max = pd.to_datetime(hist_data['nodeTime'].min()), pd.to_datetime(hist_data['nodeTime'].max())
    
    hours = datetime.timedelta(hours = 24)
    
    i = 0
    while hist_min + hours *(i + 1) <=  hist_max:
        user_ct = hist_data.loc[(hist_data['nodeTime'] >= hist_min + hours * i) 
                            & (hist_data['nodeTime'] < (hist_min + hours * (i + 1)))]['nodeUserID'].nunique()
        if user_ct != 0:
            new_users.loc[(new_users['nodeTime'] >= hist_min + hours * i) 
                      & (new_users['nodeTime'] < (hist_min + hours * (i + 1))),
                       'new_user_rate'] = (new_users.loc[(new_users['nodeTime'] >= hist_min + hours * i) 
                            & (new_users['nodeTime'] < (hist_min + hours * (i + 1)))]['nodeUserID'].nunique())/user_ct
        else:
            i +=1
        new_users.loc[(new_users['nodeTime'] >= hist_min + hours * i) & (new_users['nodeTime'] < (hist_min + hours * (i + 1))),
           'random'] = np.random.uniform(0,1,len(new_users.loc[(new_users['nodeTime'] >= hist_min + hours * i) & (new_users['nodeTime'] < (hist_min + hours * (i + 1)))]))
        i += 1
    
    #Select number of random items in new user rate list to drop for list to dataframe length agreeance  
    new_users_ls = list(new_users['new_user_rate'])
    random_ls = list(new_users['random'])
    
    new_users = data.drop_duplicates('nodeUserID')
    diff = len(new_users)-len(new_users_ls)
    
    if diff >= 0:
        new_users_ls.extend(np.random.uniform(min(new_users_ls),max(new_users_ls),diff))
        random_ls.extend(np.random.uniform(min(new_users_ls),max(new_users_ls),diff))
    elif diff < 0:
        to_delete = set(random.sample(range(len(new_users_ls)),np.abs(diff)))
        new_users_ls = [x for i,x in enumerate(new_users_ls) if i not in to_delete]

        to_delete = None
        i=None
        x=None
        
        to_delete = set(random.sample(range(len(random_ls)),np.abs(diff)))
        random_ls = [x for i,x in enumerate(random_ls) if i not in to_delete]

    #Replace new users and generate new nodeUserIDs based on assigned new user rate
    new_users['new_user_rate'] = new_users_ls
    new_users['random'] = random_ls
            
    new_users['replacement'] = new_users['nodeUserID']
    
    idx = new_users['random'] <= new_users['new_user_rate']
    new_users['replace'] = idx
    
    new_users = new_users[idx]
    new_users['replacement'] = [str(uuid.uuid4()) for i in range(len(new_users))]

    data = data.merge(new_users[['nodeUserID','replacement']],on='nodeUserID',how='left')
    data['replacement'] = data['replacement'].fillna(data['nodeUserID'])
    
    data = data.drop('nodeUserID',axis=1)
    
    data = data.rename(columns={'replacement':'nodeUserID'}) 

    for col in ['new_user_rate','random','replacement']:
        if col in data.columns:
            data = data.drop(col,axis=1)

    return(data)
        
    
def interevent_times(data, elapsed):
    #sample inter-event times from the historical data
    #draw enough samples to cover the simulation time period

    print('Sampling inter-event times...')
    #calculate inter-event times
    delta = data['nodeTime'].diff().dropna()
    
    mean_delta = delta.mean()

    n_samples = int(2.0*(elapsed/mean_delta))

    sample = delta.sample(n_samples,replace=True)
    
    #increase size of sample until it is long enough
    while sample.sum() < elapsed:
        n_samples *= 2.0
        n_samples = int(n_samples)
        sample = delta.sample(n_samples,replace=True)
        
    #cut off extra data beyond the end of the simulation period
    sample = sample[sample.cumsum() < elapsed]
    
    return(sample)

def fix_parent_relationships(sampled_df):

    #if parents occur after children in the randomly sampled data, flip their timestamps

    print('Fixing parent relationships...')
    swap_parents = True
    counter = 0
    while swap_parents:

        orig = sampled_df.copy().reset_index(drop=True)
        sampled_df = sampled_df.reset_index()

        #merge data with itself to get parent information
        parents = sampled_df[['index','nodeID','parentID']].merge(sampled_df[['index','nodeID']],how='left',
                                                                  left_on='parentID',right_on='nodeID',suffixes=('_node','_parent'))

        #find pairs where the parent occurs after the child
        parents = parents.dropna()
        parents = parents[parents['index_parent'] > parents['index_node']]
        parents = parents.drop_duplicates(subset=['parentID'])
        parents['index_parent'] = parents['index_parent'].astype(int)

        #create dictionary mapping the index of the child to the index of the parent and vice versa
        index_dict = pd.Series(parents.index_node.values,index=parents.index_parent).to_dict()
        index_dict2 = pd.Series(parents.index_parent.values,index=parents.index_node).to_dict()
        index_dict.update(index_dict2)

        #swap indices of parents and children
        sampled_df = sampled_df.rename(index=index_dict).sort_index()
        sampled_df = sampled_df.drop('index',axis=1)
        sampled_df = sampled_df.reset_index(drop=True)
        
        #if nothing was swapped this time, stop swapping
        if (orig['nodeID'] == sampled_df['nodeID']).all():
            swap_parents = False
        counter += 1
        
    return(sampled_df)
        
def sample_from_historical_data(grp, info, plat, min_time, max_time, start_time,end_time,
                                previous_hist = pd.DataFrame(columns=['nodeUserID','informationID']),
                                new_users=True):

    #add fake starting and ending event to create inter-event times from the beginning to the first
    #event and from the last event to the end of the data
    fake_start_event = pd.DataFrame({'informationID':[info],'nodeTime':[min_time],'nodeID':['-'],
                                 'nodeUserID':['-'],'parentID':['-'],'rootID':['-'],'platform':[plat],
                                     'actionType':['-'],'parentUserID':['-']})
    fake_end_event = pd.DataFrame({'informationID':[info],'nodeTime':[max_time],'nodeID':['-'],
                                   'nodeUserID':['-'],'parentID':['-'],'rootID':['-'],'platform':[plat],
                                   'actionType':['-'],'parentUserID':['-']})
    grp = pd.concat([fake_start_event,grp,fake_end_event],sort=False)

    #sample inter-event_times from the data
    delta_times = interevent_times(grp,end_time - start_time).reset_index(drop=True)
    #convert the inter-event times to actual times from the start of the simulation
    
    times = start_time + delta_times.cumsum()
    
    #remove fake events
    grp = grp[grp['nodeID'] != '-']
    #sample random events from the historical data
    sampled_df = grp.sample(len(delta_times),replace=True).reset_index(drop=True)
        
    #if duplicate events have been sampled, rename their ID fields
    sampled_df['counter'] = (sampled_df.groupby(['nodeID','parentID']).cumcount()+1).astype(str)
    sampled_df['nodeID'] = sampled_df['nodeID'] + '-' + sampled_df['counter']
    sampled_df['parentID'] = sampled_df['parentID'] + '-' + sampled_df['counter']
    sampled_df = sampled_df.drop('counter',axis=1)

    #if parents occur after children, swap them
    sampled_df = fix_parent_relationships(sampled_df)
    
    
    times.reset_index(drop=True, inplace=True)

    sampled_df['nodeTime'] = times
        
    sampled_df['nodeTime'] = pd.to_datetime(sampled_df['nodeTime'])
       
    sampled_df = sampled_df.sort_values('nodeTime')
    
    #replace some users with new users that haven't been observed before
    if new_users:
        previous_users = list(previous_hist[previous_hist['informationID'] == info]['nodeUserID'].unique())
        sampled_df = add_new_users(sampled_df,grp,previous_users = previous_users)
    
    sampled_df['nodeTime'] = pd.to_datetime(sampled_df['nodeTime'])
    sampled_df = sampled_df.sort_values('nodeTime')

    return(sampled_df)

def main():

    path = './'
    n_runs = 1
    simulation_periods = [['2020-06-01','2020-06-28']] #4 week period example

    #files are in weekly subsets, e.g. venezuela_v2_extracted_twitter_2019-02-01_2019-02-08.json
    all_files = glob.glob(path + '*.json')
    print(all_files)

    #extract dates and platforms from file names
    date_re = '(20\d\d-\d\d-\d\d)_(20\d\d-\d\d-\d\d)'
    dates = [re.search(date_re,fn) for fn in all_files]
    start_dates = [d.group(1) for d in dates]
    end_dates = [d.group(2) for d in dates]
    platforms = [re.search('twitter|youtube',fn).group(0) for fn in all_files]

    #create data frame with files, dates, and platforms
    fn_df = pd.DataFrame({'fn':all_files,
                          'start':start_dates,
                          'end':end_dates,
                          'platform':platforms})

    fn_df['start'] = pd.to_datetime(fn_df['start'])
    fn_df['end'] = pd.to_datetime(fn_df['end'])

    fn_df = fn_df.sort_values('start')

    #loop over simulation periods
    for sim_period in simulation_periods:
        #start and end time of the simulation
        start = pd.to_datetime(sim_period[0])
       
        end = pd.to_datetime(sim_period[1])
        
        #select files to sample from based on dates (same length as simulation period, just before simulation period)
        sim = fn_df[ (fn_df['start'] >= start) & (fn_df['start'] < end) ]
        hist = fn_df[ (fn_df['start'] < start) & (fn_df['start'] >= start - (end - start)) ]
        previous = fn_df[ (fn_df['start'] < start - (end - start)) ]
        
        print(start,end)
        print('Historical Data to Sample From')
        print(hist)
        print('Prior Data to Track Users From')
        print(previous)
        
        previous_history_data = list(previous['fn'].values)
        history_data = list(hist['fn'].values)

        #load data
        hist = []
        for data in history_data:
            hist.append(ss.load_data(data, ignore_first_line=False, verbose=False))
        hist = pd.concat(hist)

        hist = hist.sort_values('nodeTime')
        
        #Get parentUserID here. When working with CP4 data, add parentUserID column.
        #hist['parentUserID'] = [str(uuid.uuid4()) for i in range(len(hist))] and ,'parentUserID'
        hist = hist[['informationID','nodeTime','nodeID','parentID','rootID','platform','actionType','nodeUserID','parentUserID']]

        #For extracted ground truth files containing all narratives outside of the 18 core
        node_ls =  ['controversies/pakistan/students',
                   'leadership/sharif','leadership/bajwa',
                   'controversies/china/uighur',
                   'controversies/china/border',
                   'benefits/development/roads',
                   'controversies/pakistan/baloch',
                   'benefits/jobs',
                   'opposition/propaganda',
                   'benefits/development/energy',
                   'controversies/pakistan/bajwa',
                    'other']                   

        
        hist = hist.loc[hist['informationID'].isin(node_ls)]

        previous_hist = []
        for data in previous_history_data:
            previous_hist.append(ss.load_data(data, ignore_first_line=False, verbose=False))
        previous_hist = pd.concat(previous_hist)
        previous_hist = previous_hist[['nodeUserID','informationID']].drop_duplicates()

        previous_hist = previous_hist.loc[previous_hist['informationID'].isin(node_ls)]
        
     
        #multiple runs of the baseline sampling
        for i in range(n_runs):
            dfs = []
            # for each platform and information ID
            for (plat,info),grp in hist.groupby(['platform','informationID']):
                

                print(plat,info)

                starting = time.time()
                sampled_df = sample_from_historical_data(grp, info, plat,
                                                         hist['nodeTime'].min(), hist['nodeTime'].max(),
                                                         start,end + datetime.timedelta(days=1),
                                                         previous_hist = previous_hist,
                                                         new_users=True)
                ending = time.time()
                elapsed = (ending - starting)/60.0
                print(f'Time elapsed: {elapsed} minutes')

                dfs.append(sampled_df)

            baseline = pd.concat(dfs).reset_index(drop=True)
            
            #save generated baseline
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')

            baseline.to_json(f'/u00/SocialSim/home/millersa4/baseline/baseline_{start_str}_{end_str}_{i}.json',orient='records',lines=True)
            
            
            start_str_meta = start.strftime('%B%d').lower()
            end_str_meta = end.strftime('%B%d').lower()

            submission_meta = '{"team": "leidos", "model_identifier": "event_sampled_baseline", "simulation_period": ' + f'"{start_str_meta}-{end_str_meta}"' + '}\n'

            with open(f'/u00/SocialSim/home/millersa4/baseline/baseline_{start_str}_{end_str}_{i}.json', 'r+') as fp:
                lines = fp.readlines()     # lines is list of line, each element '...\n'
                lines.insert(0, submission_meta)  # you can use any index if you know the line index
                fp.seek(0)                 # file pointer locates at the beginning to write the whole file again
                fp.writelines(lines)       # write whole lists again to the same file


    
if __name__ == '__main__':
   main()
