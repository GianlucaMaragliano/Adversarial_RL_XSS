

from utils.utils import init_argument_parser
from utils.path_utils import get_subfolders
import os
import json

def avg_adversarial_results(runs_folder, model, vocab_size, detector_run, adv_folder, adv_results_file, ruin_rates_file):
    adversarial_agents_runs = list(map(lambda x: x.split('/')[-1],get_subfolders(os.path.join(runs_folder,model,vocab_size,detector_run,adv_folder))))
    results = dict()
    run_results = []
    for run in adversarial_agents_runs:
        with open(os.path.join(runs_folder,model,vocab_size,detector_run,adv_folder,run,adv_results_file)) as f:
            results_adv = json.load(f)
        with open(os.path.join(runs_folder,model,vocab_size,detector_run,adv_folder,run,ruin_rates_file)) as f:
            ruin_rate = json.load(f)
        #merge the two dicts into run_result
        run_result = {**results_adv, **ruin_rate}
        run_results.append(run_result)
    #average the results
    for key in run_results[0].keys():
        results[key] = sum([run[key] for run in run_results])/len(run_results)
    return results

def summarize_results(opt):
    results = dict()
    models = list(map(lambda x: x.split('/')[-1],get_subfolders(opt.runs_folder)))
    for model in models:
        results[model] = dict()
        detector_vocab_sizes = list(map(lambda x: x.split('/')[-1],get_subfolders(os.path.join(opt.runs_folder,model))))
        for vocab_size in detector_vocab_sizes:
            results[model][vocab_size] = dict()
            detector_runs = list(map(lambda x: x.split('/')[-1],get_subfolders(os.path.join(opt.runs_folder,model,vocab_size))))

            for detector_run in detector_runs:
            
                #load json with test_results
                with open(os.path.join(opt.runs_folder,model,vocab_size,detector_run,opt.detector_results_file)) as f:
                    results[model][vocab_size][detector_run]= json.load(f)
                results[model][vocab_size][detector_run]["adversarial_agent"] = avg_adversarial_results(opt.runs_folder, model, vocab_size,detector_run, opt.adversarial_agent_folder, opt.adv_results_file, opt.ruin_rates_file)
                
                results[model][vocab_size][detector_run]["adversarial_agent_oracle_guided"] = avg_adversarial_results(opt.runs_folder, model, vocab_size, detector_run, opt.adversarial_agent_oracle_guided_folder, opt.adv_results_file, opt.ruin_rates_file)

    with open(os.path.join(opt.runs_folder, opt.summary_file), 'w') as f:
        json.dump(results, f, indent=4)
def add_parse_arguments(parser):

    parser.add_argument('--runs_folder', type=str, default = "runs", help='Folder with the runs')
    parser.add_argument('--adversarial_agent_folder', type=str, default = "adversarial_agent", help='Subfolder with the adversarial agents')
    parser.add_argument('--adversarial_agent_oracle_guided_folder', type=str, default = "adversarial_agent_oracle_guided", help='Subfolder with the oracle guided adversarial agents')

    parser.add_argument('--summary_file', type=str, default="summary.json", help='File with the summary of the results')
    parser.add_argument('--detector_results_file', type=str, default="test_results.json", help='File with the results of the detector')
    parser.add_argument('--adv_results_file', type=str, default="results.json", help='File with the results of the adversarial agent')
    parser.add_argument('--ruin_rates_file', type=str, default="ruin_rate.json", help='File with the ruin rates of the adversarial agent')

    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    summarize_results(opt)

if __name__ == '__main__':
    main()