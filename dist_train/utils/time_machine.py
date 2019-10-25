# import os
# import json
# import shutil
# import argparse
# from tqdm import tqdm
# from dist_train.utils.dirs import BASE_DIR
#
#
# def remove_timestamped_files(exp_dir, cutoff):
#     print('Removing timestamped files...', flush=True)
#     contents = os.listdir(exp_dir)
#     n_removed = 0
#     for thing in tqdm(contents):
#         path = os.path.join(exp_dir, thing)
#         if not os.path.isfile(path):
#             continue
#         prefix = thing.split('_')[0]
#         prefix = prefix.split('.')[0]
#
#         if not prefix.isdigit():
#             continue
#
#         if int(prefix) > cutoff:
#             os.remove(path)
#             n_removed += 1
#     print('Removed {} timestamped files.\n'.format(n_removed), flush=True)
#
#
# def roll_back_logs(exp_dir, cutoff):
#     def roll_back_log_type(log_type):
#         log_type = log_type.lower()
#         print('Removing unwanted episodes from the {} files...'.format(log_type), flush=True)
#         contents = os.listdir(exp_dir)
#         log_paths = []
#         for thing in contents:
#             path = os.path.join(exp_dir, thing)
#             if not os.path.isfile(path):
#                 continue
#             prefix = thing.split('_')[0]
#
#             if prefix == log_type:
#                 log_paths.append(path)
#
#         n_cleared = 0
#         for path in log_paths:
#             print('   Cleaning up {}'.format(path), flush=True)
#             with open(path) as f:
#                 tmp = f.read()
#
#             with open(os.path.join(exp_dir, 'tmp.json'), 'a') as save_file:
#                 for kv in tmp[1:-1].split('}{'):
#                     kvs = kv.split(':')
#                     key = kvs[0]
#                     v = ''.join(kvs[1:])
#                     try:
#                         k = eval(key)
#                         v = eval(v)
#                         ep_number = int(float(k))
#                         if ep_number < cutoff:
#                             save_file.write(json.dumps({k: v}))
#                         else:
#                             n_cleared += 1
#                     except SyntaxError:
#                         n_cleared += 1
#                         continue
#                     except:
#                         raise
#                 save_file.close()
#
#             os.remove(path)
#             shutil.copyfile(os.path.join(exp_dir, 'tmp.json'), path)
#             os.remove(os.path.join(exp_dir, 'tmp.json'))
#         print('Cleared a total of {} {} entries.\n'.format(n_cleared, log_type), flush=True)
#
#     roll_back_log_type('hist')
#     roll_back_log_type('log')
#
#
# def set_leading_snapshot(exp_dir, cutoff):
#     print('Setting leading snapshot...', flush=True)
#     contents = os.listdir(exp_dir)
#     for thing in contents:
#         path = os.path.join(exp_dir, thing)
#         if not os.path.isfile(path):
#             continue
#         if path[-7:] == 'pth.tar':
#             name_parts = thing.split('_')
#             if len(name_parts) < 2:
#                 continue
#             prefix = name_parts[0]
#             name = name_parts[1]
#
#             if not prefix.isdigit():
#                 continue
#
#             if int(prefix) == cutoff:
#                 print('   Set {} as leading'.format(thing), flush=True)
#                 shutil.copyfile(path, os.path.join(exp_dir, name))
#
#
# def determine_cutoff(exp_dir, last_good):
#     contents = os.listdir(exp_dir)
#     max_n = -1
#     for thing in contents:
#         path = os.path.join(exp_dir, thing)
#         if not os.path.isfile(path):
#             continue
#
#         name_parts = thing.split('_')
#         if len(name_parts) < 2:
#             continue
#
#         prefix = name_parts[0]
#         name = name_parts[1]
#
#         if not prefix.isdigit():
#             continue
#
#         if name == 'model.pth.tar':
#             n = int(prefix)
#             if (n > max_n) and (n <= last_good):
#                 max_n = int(prefix)
#
#     return max_n
#
#
# def roll_back(tag, last):
#     experiment_directory = os.path.join(BASE_DIR, tag)
#     episode_cutoff = determine_cutoff(experiment_directory, last)
#
#     if episode_cutoff < 0:
#         return
#
#     print('\nGoing back in time to episode number {}\n'.format(episode_cutoff), flush=True)
#
#     remove_timestamped_files(experiment_directory, episode_cutoff)
#     roll_back_logs(experiment_directory, episode_cutoff)
#     set_leading_snapshot(experiment_directory, episode_cutoff)
#
#     print('\nTime machine successful!', flush=True)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("Revert an experiment to the last snapshot before a given episode count.")
#
#     # Required arguments
#     parser.add_argument('tag', type=str,
#                         help='Experiment subdirectory tag')
#
#     parser.add_argument('last', type=int,
#                         help='Last episode where the model was in a good state')
#
#     args = parser.parse_args()
#
#     roll_back(args.tag, args.last)
#
#
#
