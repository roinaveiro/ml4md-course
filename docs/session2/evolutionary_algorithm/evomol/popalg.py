import copy
import csv
import time
from collections import deque
from os import makedirs
from os.path import dirname, join

import numpy as np
from rdkit.Chem.rdmolfiles import MolFromSmiles

from .evaluation import EvaluationError, scores_to_scores_dict
from .molgraphops.molgraph import MolGraph
from .mutation import NoImproverError, MutationError


class NoMoreIndToMutate(Exception):
    pass


class GeneratedIndividualsRecorder:

    def __init__(self, curr_step, evaluation_strategy):
        """
        Object used to record all generated solutions (even if not inserted) during a step
        """

        self.curr_step = curr_step
        self.smiles = []
        self.total_scores = []
        self.scores = []
        self.objective_calls = []
        self.success_obj_computation = []
        self.improver = []
        self.failed_any_filter = []
        self.failed_any_quality_filter = []
        self.failed_tabu_pop = []
        self.failed_tabu_external = []
        self.failed_rdfilters = []
        self.failed_sillywalks = []
        self.failed_sascore = []
        self.failed_custom_filter = []
        self.evaluation_strategy = evaluation_strategy
        self.obj_computation_time = []

    def record_individual(self, individual, total_score, scores, objective_calls, improver, success_obj_computation,
                          obj_computation_time, failed_tabu_pop=False, failed_tabu_external=False,
                          failed_rdfilters=False, failed_sillywalks=False, failed_sascore=False,
                          failed_custom_filter=False):
        self.smiles.append(individual.to_aromatic_smiles())
        self.total_scores.append(total_score)
        self.scores.append(scores)
        self.objective_calls.append(objective_calls)
        self.improver.append(improver)
        self.success_obj_computation.append(success_obj_computation)
        self.obj_computation_time.append(obj_computation_time)
        failed_any_quality_filter = failed_rdfilters or failed_sillywalks or failed_sascore or failed_custom_filter
        self.failed_any_quality_filter.append(failed_any_quality_filter)
        self.failed_any_filter.append(failed_any_quality_filter or failed_tabu_pop or failed_tabu_external)
        self.failed_tabu_pop.append(failed_tabu_pop)
        self.failed_tabu_external.append(failed_tabu_external)
        self.failed_rdfilters.append(failed_rdfilters)
        self.failed_sillywalks.append(failed_sillywalks)
        self.failed_sascore.append(failed_sascore)
        self.failed_custom_filter.append(failed_custom_filter)

    def get_scores_array(self):
        return np.array(self.scores).reshape(len(self.smiles), len(self.evaluation_strategy.keys()))

    def get_step_vect(self):
        return list(np.full((len(self.smiles),), self.curr_step))

    def get_passed_filters_mask(self):
        return np.logical_not(self.failed_any_filter)


class PopAlg:
    """
    Class running the population algorithm defined by the given strategies.
    """

    def copy_instance_with_parameters(self):
        """
        Copying the instance with (only) the parameters. The strategies must be set for each copied instance.
        Allows parallelism.
        """

        return PopAlg(
            evaluation_strategy=copy.deepcopy(self.evaluation_strategy),
            mutation_strategy=copy.deepcopy(self.mutation_strategy),
            stop_criterion_strategy=copy.deepcopy(self.stop_criterion_strategy),
            pop_max_size=self.pop_max_size,
            k_to_replace=self.k_to_replace,
            save_n_steps=self.save_n_steps,
            print_n_steps=self.print_n_steps,
            kth_score_to_record=self.kth_score_to_record,
            record_history=self.record_history,
            problem_type=self.problem_type,
            selection=self.selection,
            kth_score_to_record_key=self.kth_score_to_record_key,
            shuffle_init_pop=self.shuffle_init_pop,
            sulfur_valence=self.sulfur_valence,
            external_tabu_list=self.external_tabu_list,
            record_all_generated_individuals=self.record_all_generated_individuals
        )

    def __init__(self, evaluation_strategy, mutation_strategy, stop_criterion_strategy,
                 output_folder_path="EvoMol_model/", pop_max_size=1000, k_to_replace=10, save_n_steps=100,
                 print_n_steps=1, kth_score_to_record=1, record_history=False, problem_type="max", selection="best",
                 kth_score_to_record_key="total", shuffle_init_pop=False, external_tabu_list=None,
                 record_all_generated_individuals=False, evaluation_strategy_parameters=None, sulfur_valence=6):
        """
        :param evaluation_strategy: EvaluationStrategy instance to evaluate individuals
        :param mutation_strategy: MutationStrategy instance to mutate solutions and find improvers
        :param stop_criterion_strategy: StopCriterionStrategy instance to stop the search when a condition is reached
        :param output_folder_path: Path of folder where the data is recorded (default : "EvoMol_model/")
        :param pop_max_size: Maximum population size (default : 1000)
        :param k_to_replace: Number of individuals to be replaced each step (default : 10)
        :param save_n_steps: Frequency of saving the model (default : 100)
        :param print_n_steps: Frequency of printing the results (default : 1)
        :param kth_score_to_record: Kth score to be recorded for premature stop
        :param record_history: Whether to record history of actions (necessary to draw exploration trees)
        :param problem_type: Whether it is a maximization ("max") or minimization ("min") problem. (default : "max")
        :param selection: Whether best individuals are selected ("best") to be mutated or individuals are selected
        randomly ("random"). (default : "best")
        :param kth_score_to_record_key: string key of the kth best score to be recorded dynamically
        :param shuffle_init_pop: whether to shuffle initial population at initialization
        :param external_tabu_list: list of SMILES that cannot be generated by EvoMol
        :param record_all_generated_individuals: whether to record all individuals that are generated in a file
        along with the number of calls to the objective function at the time of insertion. Also recording individuals
        that failed the objective computation
        :param evaluation_strategy_parameters: allows to set evaluation_strategy parameters depending on context.
        Available contexts are "evaluate_new_solution" and "evaluate_init_pop"
        :param sulfur_valence: maximum valence of sulfur atoms (default : 6)
        """

        # Loading problem type
        self.problem_type = problem_type

        # Loading strategy modules
        self.evaluation_strategy = evaluation_strategy
        self.mutation_strategy = mutation_strategy
        self.stop_criterion_strategy = stop_criterion_strategy

        # Updating PopAlg instance in StopCriterionStrategy instance
        self.stop_criterion_strategy.set_pop_alg_instance(self)

        # Saving population's max size and the number of individuals to replace each step
        self.pop_max_size = pop_max_size
        self.k_to_replace = k_to_replace

        # Frequency of saving and printing results
        self.save_n_steps = save_n_steps
        self.print_n_steps = print_n_steps

        # Recording output files paths
        self.output_folder_path = output_folder_path

        # History recording parameter
        self.record_history = record_history

        # All inserted individuals recording parameter
        self.record_all_generated_individuals = record_all_generated_individuals

        # Selection parameter
        self.selection = selection

        # Kth score to record key
        self.kth_score_to_record_key = kth_score_to_record_key

        # Kth score to record
        self.kth_score_to_record = kth_score_to_record

        # Whether to shuffle initial population
        self.shuffle_init_pop = shuffle_init_pop

        # Attributes initialization
        self.pop = None
        self.all_generated_individuals_smiles = None
        self.all_generated_individuals_n_obj_calls = None
        self.all_generated_individuals_step = None
        self.all_generated_individuals_obj_value = None
        self.all_generated_individuals_scores = None
        self.all_generated_individuals_improver = None
        self.all_generated_individuals_success_obj_computation = None
        self.all_generated_individuals_obj_computation_time = None
        self.pop_tabu_list = None
        self.external_tabu_list = external_tabu_list
        self.step_traces = None
        self.curr_step_id = None
        self.errors = None
        self.curr_total_scores = None
        self.curr_scores = None
        self.kth_score_history = None
        self.n_success_mut = None
        self.n_fail_mut = None
        self.actions_history = None
        self.removed_actions_score_smi_tuple = None
        self.timestamp_start = None
        self.kth_score_to_record_idx = None

        self.evaluation_strategy_parameters = {
            "evaluate_new_solution": {},
            "evaluate_init_pop": {}
        } if evaluation_strategy_parameters is None else evaluation_strategy_parameters

        self.sulfur_valence = sulfur_valence

    def initialize(self):
        """
        Initialization of EvoMol with starting values.
        This method MUST BE CALLED BEFORE running the algorithm.
        :return:
        """

        # Initialization of population
        self.pop = list(np.full((self.pop_max_size,), None))

        # Initialization of the dictionaries containing the smiles of former and current individuals as keys
        self.pop_tabu_list = list(np.full((self.pop_max_size,), None))

        # Intialization of the list of all individual ever inserted in the population, the list of their
        # corresponding number of calls to the objective function at insertion and the list of the corresponding steps.
        # Also recording the values of the objective function
        self.all_generated_individuals_smiles = []
        self.all_generated_individuals_n_obj_calls = []
        self.all_generated_individuals_step = []
        self.all_generated_individuals_obj_value = []
        self.all_generated_individuals_scores = np.array([]).reshape(0, len(self.evaluation_strategy.keys()))
        self.all_generated_individuals_improver = []
        self.all_generated_individuals_success_obj_computation = []
        self.all_generated_individuals_obj_computation_time = []

        # Insuring the SMILES of the external tabu list are canonical
        if self.external_tabu_list is not None:
            self.external_tabu_list = [MolGraph(MolFromSmiles(smi)).to_aromatic_smiles() for smi in
                                       self.external_tabu_list]

        # Initialization of the dictionary containing the traces of steps of the algorithm
        self.step_traces = {
            'scores': {},
            'n_replaced': [],
            'additional_values': {},
            'timestamps': [],
            'n_failed_obj_computation': [],
            'n_not_improvers_among_success_obj_computation': [],
            'n_discarded_tabu': [],
            'n_discarded_filters': [],
            'n_discarded_rdfilters': [],
            'n_discarded_sillywalks': [],
            'n_discarded_sascore': [],
            'n_discarded_custom_filter': []
        }

        # Initialization of keys in the self.step_traces dict declared by the evaluation strategy instance
        for k in self.evaluation_strategy.keys() + ["total"]:
            for stat in ["mean", "med", "min", "max", "std"]:
                self.step_traces["scores"][k + "_" + stat] = []

        # Initialization of keys in the self.step_traces dict for additional population scores
        for k in self.evaluation_strategy.get_additional_population_scores().keys():
            print(k)
            self.step_traces['additional_values'][k] = []

        # Initialization of the step counter.
        self.curr_step_id = 0

        # Initialization of errors list
        self.errors = []

        self.curr_total_scores = None
        self.curr_scores = None
        self.timestamp_start = None

        # Computing idx of kth score to be recorded vector
        for i, k in enumerate(self.evaluation_strategy.keys()):
            if k == self.kth_score_to_record_key:
                self.kth_score_to_record_idx = i

        self.kth_score_history = deque(maxlen=500)

        self.n_success_mut = np.zeros(self.pop_max_size, dtype=np.int)
        self.n_fail_mut = np.zeros(self.pop_max_size, dtype=np.int)

        self.actions_history = list(np.full(self.pop_max_size, None))
        self.removed_actions_score_smi_tuple = {}

        # Computing start timestamp
        self.timestamp_start = time.time()

    def load_pop_from_smiles_list(self, smiles_list, atom_mutability=True):
        """
        Loading the population from the given smiles list.
        Setting the internal variables to their values
        :param smiles_list: list of SMILES
        :param atom_mutability: whether the core of the molecules of the starting population can be modified
        :return:
        """

        if self.shuffle_init_pop:
            np.random.shuffle(smiles_list)

        # Iterating over all the given smiles
        for i, smi in enumerate(smiles_list):
            # Loading QuMolGraph object
            self.pop[i] = MolGraph(MolFromSmiles(smi), sanitize_mol=True, mutability=atom_mutability,
                                   sulfur_valence=self.sulfur_valence)

            # Saving smiles in the tabu dictionary and in action history initialization
            self.pop_tabu_list[i] = self.pop[i].to_aromatic_smiles()
            self.actions_history[i] = self.pop[i].to_aromatic_smiles()

        # Evaluation of the population (not recording the count of calls to the objective function)
        print("Computing scores at initialization...")
        self.evaluation_strategy.set_params(**self.evaluation_strategy_parameters["evaluate_init_pop"])
        self.evaluation_strategy.disable_calls_count()
        self.evaluation_strategy.compute_record_scores_init_pop(self.pop)
        self.evaluation_strategy.enable_calls_count()
        self.evaluation_strategy.set_params(**self.evaluation_strategy_parameters["evaluate_new_solution"])

        # Recording the scores of the initial population in the all_generated.csv file.
        # The file is thus badly named indeed. But it will stay that way for compatibility reasons.
        if self.record_all_generated_individuals:
            scores, all_scores = self.evaluation_strategy.get_population_scores()
            comput_time = self.evaluation_strategy.get_population_comput_time_vector()

            # Iterating over all individuals of the initial population
            for i in range(len(scores)):
                self.all_generated_individuals_smiles.append(self.pop[i].to_aromatic_smiles())
                self.all_generated_individuals_improver.append(None)
                self.all_generated_individuals_step.append(-1)
                self.all_generated_individuals_n_obj_calls.append(0)
                self.all_generated_individuals_success_obj_computation.append(True)
                self.all_generated_individuals_obj_value.append(scores[i])
                self.all_generated_individuals_obj_computation_time.append(comput_time[i])

                all_scores_vect = []
                for j in range(len(self.evaluation_strategy.keys())):
                    all_scores_vect.append(all_scores[j][i])
                print(all_scores_vect)
                self.all_generated_individuals_scores = np.concatenate([self.all_generated_individuals_scores,
                                                                        np.array(all_scores_vect).reshape(1, -1)])

    def save(self):
        """
        Saving the data to the files
        :return:
        """
        if self.output_folder_path:

            # Creating directories if they don't exist
            makedirs(dirname(join(self.output_folder_path, "file")), exist_ok=True)

            # ### Steps data ###
            csv_array = []
            for k, v in self.step_traces["scores"].items():
                csv_array.append([k] + v)
            for k, v in self.step_traces["additional_values"].items():
                csv_array.append([k] + v)
            csv_array.append(["n_replaced"] + self.step_traces["n_replaced"])
            csv_array.append(["timestamps"] + self.step_traces["timestamps"])
            csv_array.append(["n_failed_obj_computation"] + self.step_traces["n_failed_obj_computation"])
            csv_array.append(["n_not_improvers_among_success_obj_computation"] +
                             self.step_traces["n_not_improvers_among_success_obj_computation"])
            csv_array.append(["n_discarded_tabu"] + self.step_traces["n_discarded_tabu"])
            csv_array.append(["n_discarded_filters"] + self.step_traces["n_discarded_filters"])
            csv_array.append(["n_discarded_rdfilters"] + self.step_traces["n_discarded_rdfilters"])
            csv_array.append(["n_discarded_sillywalks"] + self.step_traces["n_discarded_sillywalks"])
            csv_array.append(["n_discarded_sascore"] + self.step_traces["n_discarded_sascore"])
            csv_array.append(["n_discarded_custom_filter"] + self.step_traces["n_discarded_custom_filter"])

            with open(join(self.output_folder_path, 'steps.csv'), "w", newline='') as f:
                writer = csv.writer(f)
                for row in np.array(csv_array).T:
                    writer.writerow(row)

            # ### All inserted individuals data ###
            if self.record_all_generated_individuals:
                csv_array = [["step"] + self.all_generated_individuals_step,
                             ["SMILES"] + self.all_generated_individuals_smiles,
                             ["obj_calls"] + self.all_generated_individuals_n_obj_calls,
                             ["obj_value"] + self.all_generated_individuals_obj_value,
                             ["improver"] + self.all_generated_individuals_improver,
                             ["success_obj_computation"] + self.all_generated_individuals_success_obj_computation,
                             ["obj_computation_time"] + self.all_generated_individuals_obj_computation_time]

                for i, key in enumerate(self.evaluation_strategy.keys()):

                    csv_array.append(
                        [key] + self.all_generated_individuals_scores.T[i].tolist()
                    )

                with open(join(self.output_folder_path, "all_generated.csv"), "w") as f:
                    writer = csv.writer(f)
                    for row in np.array(csv_array).T:
                        writer.writerow(row)

            # ### Last step population data ###
            csv_array = []

            # Mutation success history
            n_success_mut_str = []
            n_fail_mut_str = []
            for i, ind in enumerate(self.pop):
                n_success_mut_str.append(str(self.n_success_mut[i]))
                n_fail_mut_str.append(str(self.n_fail_mut[i]))

            csv_array.append(["smiles"] + self.pop_tabu_list)

            # Mutation success and failures
            csv_array.append(["n_success_mut"] + n_success_mut_str)
            csv_array.append(["n_failures_mut"] + n_fail_mut_str)

            # Scores data
            self.curr_total_scores, self.curr_scores = self.evaluation_strategy.get_population_scores()
            step_scores_dict = scores_to_scores_dict(self.curr_total_scores,
                                                     self.curr_scores,
                                                     self.evaluation_strategy.keys())

            for k, scores_list in step_scores_dict.items():
                scores_list_np = np.full((self.pop_max_size,), None)
                scores_list_np[:len(scores_list)] = scores_list
                csv_array.append([k] + list(scores_list_np))

            # Computation time
            obj_comput_time_list = list(self.evaluation_strategy.get_population_comput_time_vector())
            obj_comput_time_np = np.full((self.pop_max_size,), None)
            obj_comput_time_np[:len(obj_comput_time_list)] = obj_comput_time_list
            csv_array.append(["obj_computation_time"] + list(obj_comput_time_np))

            # Action history data
            csv_array.append(["history_data"] + self.actions_history)

            with open(join(self.output_folder_path, 'pop.csv'), "w", newline='') as f:
                writer = csv.writer(f)
                for row in np.array(csv_array).T:
                    writer.writerow(row)

            # ### Removed individuals actions recording ###
            if self.record_history:
                with open(join(self.output_folder_path, 'removed_ind_act_history.csv'), "w", newline='') as f:

                    writer = csv.writer(f)
                    writer.writerow(["history_data", "total"] + self.evaluation_strategy.keys() + ["smiles"])

                    for removed_act_history in self.removed_actions_score_smi_tuple.keys():
                        if removed_act_history != "":
                            total_score = self.removed_actions_score_smi_tuple[removed_act_history][0]
                            scores = self.removed_actions_score_smi_tuple[removed_act_history][1]
                            smi = self.removed_actions_score_smi_tuple[removed_act_history][2]

                            writer.writerow([removed_act_history, total_score] + list(scores) + [smi])

            # ### Errors data ###
            with open(join(self.output_folder_path, 'errors.csv'), "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "error"])
                for error in self.errors:
                    writer.writerow(error)

    def record_step_data(self, step_gen_ind_recorder):
        """
        :param step_gen_ind_recorder: GeneratedIndividualsRecorder instance to retrieve information about discarded
        solutions
        :return:
        """

        # Extracting scores dictionary containing the scores for each objective
        step_scores_dict = scores_to_scores_dict(self.curr_total_scores, self.curr_scores,
                                                 self.evaluation_strategy.keys())

        # Saving statistics on current step scores
        for k, scores_list in step_scores_dict.items():
            step_mean = np.mean(scores_list)
            step_min = np.min(scores_list)
            step_max = np.max(scores_list)
            step_med = np.median(scores_list)
            step_std = np.std(scores_list)

            self.step_traces["scores"][k + "_mean"].append(step_mean)
            self.step_traces["scores"][k + "_med"].append(step_med)
            self.step_traces["scores"][k + "_min"].append(step_min)
            self.step_traces["scores"][k + "_max"].append(step_max)
            self.step_traces["scores"][k + "_std"].append(step_std)

            if self.curr_step_id % self.print_n_steps == 0:
                print(k + "_mean : " + str("%.5f" % step_mean))
                print(k + "_med : " + str("%.5f" % step_med))
                print(k + "_std : " + str("%.5f" % step_std))
                print(k + "_min : " + str("%.5f" % step_min))
                print(k + "_max : " + str("%.5f" % step_max))

        # Extracting additional scores of population for current step from the evaluator
        for k, v in self.evaluation_strategy.get_additional_population_scores().items():
            self.step_traces["additional_values"][k].append(v)

        # Saving information about discarded solutions during the step
        if step_gen_ind_recorder is not None:
            self.step_traces["n_failed_obj_computation"].append(np.sum(
                np.logical_not(np.array(step_gen_ind_recorder.success_obj_computation)[
                                   step_gen_ind_recorder.get_passed_filters_mask()])))
            self.step_traces["n_not_improvers_among_success_obj_computation"].append(np.sum(
                np.logical_not(step_gen_ind_recorder.improver)[np.logical_and(
                    np.array(step_gen_ind_recorder.success_obj_computation),
                    step_gen_ind_recorder.get_passed_filters_mask())]
            ))
            self.step_traces["n_discarded_tabu"].append(np.sum(np.logical_or(
                step_gen_ind_recorder.failed_tabu_pop,
                step_gen_ind_recorder.failed_tabu_external
            )))
            self.step_traces["n_discarded_filters"].append(np.sum(step_gen_ind_recorder.failed_any_quality_filter))
            self.step_traces["n_discarded_rdfilters"].append(np.sum(step_gen_ind_recorder.failed_rdfilters))
            self.step_traces["n_discarded_sillywalks"].append(np.sum(step_gen_ind_recorder.failed_sillywalks))
            self.step_traces["n_discarded_sascore"].append(np.sum(step_gen_ind_recorder.failed_sascore))
            self.step_traces["n_discarded_custom_filter"].append(np.sum(step_gen_ind_recorder.failed_custom_filter))
        else:
            self.step_traces["n_failed_obj_computation"].append(0)
            self.step_traces["n_not_improvers_among_success_obj_computation"].append(0)
            self.step_traces["n_discarded_tabu"].append(0)
            self.step_traces["n_discarded_filters"].append(0)
            self.step_traces["n_discarded_rdfilters"].append(0)
            self.step_traces["n_discarded_sillywalks"].append(0)
            self.step_traces["n_discarded_sascore"].append(0)
            self.step_traces["n_discarded_custom_filter"].append(0)

    def evaluate_pop_record_step_data(self, n_replaced, record_step_data=True, step_gen_ind_recorder=None):

        # Population evaluation
        self.curr_total_scores, self.curr_scores = self.evaluation_strategy.get_population_scores()

        # Extracting vector of scores corresponding to the key of the kth score to be recorded
        if self.kth_score_to_record_key == "total":
            scores_vector_kth_score_to_be_recorded = self.curr_total_scores
        else:
            scores_vector_kth_score_to_be_recorded = self.curr_scores[self.kth_score_to_record_idx]

        # Updating the history of the kth score
        if len(scores_vector_kth_score_to_be_recorded) >= self.kth_score_to_record:
            kth_score = np.partition(scores_vector_kth_score_to_be_recorded, -self.kth_score_to_record)[
                -self.kth_score_to_record]
            self.kth_score_history.appendleft(kth_score)
        else:
            self.kth_score_history.appendleft(np.nan)

        if record_step_data:
            # Recording step data
            self.record_step_data(step_gen_ind_recorder)

            # Saving step timestamp
            self.step_traces["timestamps"].append(time.time() - self.timestamp_start)

            # Saving the number of replaced individuals
            self.step_traces["n_replaced"].append(n_replaced)

    def select_to_be_replaced(self):

        # Computing number of defined individuals
        n_defined_ind = len(self.curr_total_scores)

        # Computing priority order of undefined individuals
        undefined_priority_order = np.arange(n_defined_ind, self.pop_max_size)

        defined_priority_order = None

        if self.problem_type == "max":
            defined_priority_order = np.argsort(self.curr_total_scores)
        elif self.problem_type == "min":
            defined_priority_order = np.argsort(self.curr_total_scores)[::-1]

        # Computing complete order
        to_be_replaced_indices = list(undefined_priority_order) + list(defined_priority_order)

        return to_be_replaced_indices[:self.k_to_replace]

    def sort_to_be_mutated(self):

        # Computing max of valid individual (that are defined)
        ind_valid_idx = []

        for i, ind in enumerate(self.pop):
            if ind is not None:
                ind_valid_idx.append(i)

        ind_valid_idx = np.array(ind_valid_idx)

        # Extracting the scores of the valid individual
        scores_valid = np.array(self.curr_total_scores)[ind_valid_idx]

        to_be_mutated_in_order_mask = None

        # Sorting in descending order the scores of the valid individuals if selection of best individuals
        if self.selection == "best":
            if self.problem_type == "max":
                to_be_mutated_in_order_mask = ind_valid_idx[np.argsort(scores_valid)[::-1].flatten()]
            elif self.problem_type == "min":
                to_be_mutated_in_order_mask = ind_valid_idx[np.argsort(scores_valid).flatten()]

        # Selecting the defined individuals randomly
        elif self.selection == "random":
            np.random.shuffle(ind_valid_idx)
            to_be_mutated_in_order_mask = ind_valid_idx

        # Selecting the individuals according to a random strategy with probabilities that are proportional to the value
        elif self.selection == "random_weighted":

            # If the problem is a minimization problem then computing the opposite of scores
            if self.problem_type == "min":
                scores_valid = -scores_valid

            # If negative scores exist, shifting all values so that minimum is 0
            if np.min(scores_valid) < 0:
                scores_valid_shifted = scores_valid - np.min(scores_valid)
            else:
                scores_valid_shifted = scores_valid

            # Setting all zero objective values to a small number so that probability computation can be performed
            scores_valid_shifted[scores_valid_shifted == 0] = 1e-10

            to_be_mutated_in_order_mask = np.random.choice(ind_valid_idx, len(ind_valid_idx), replace=False,
                                                           p=scores_valid_shifted / scores_valid_shifted.sum())

        return to_be_mutated_in_order_mask

    def get_k_best_individuals_smiles(self, k_best, tabu_list=None):
        """
        Returning the k best individuals of the population that are not in the given tabu list of SMILES (if specified).
        :param k_best: number of individuals to return
        :param tabu_list: list of SMILES that cannot be returned
        :return: list of SMILES, list of scores, list of sub-scores
        """

        scores_sort = np.argsort(self.curr_total_scores)

        if self.problem_type == "max":
            scores_sort = scores_sort[::-1]

        best_solutions_ind = np.array(self.pop)[scores_sort]
        best_solutions_scores = self.curr_total_scores[scores_sort]
        best_solutions_sub_scores = self.curr_scores.T[scores_sort]

        returned_smiles = []
        returned_scores = []
        returned_sub_scores = []
        i = 0
        # Computing the list of k_best best individuals that are not in the tabu list
        while len(returned_smiles) < k_best and i < len(best_solutions_ind):
            curr_sol_smiles = MolGraph(MolFromSmiles(best_solutions_ind[i].to_aromatic_smiles())).to_aromatic_smiles()
            if tabu_list is None or curr_sol_smiles not in tabu_list:
                returned_smiles.append(curr_sol_smiles)
                returned_scores.append(best_solutions_scores[i])
                returned_sub_scores.append(best_solutions_sub_scores[i])

            i += 1

        return returned_smiles, returned_scores, returned_sub_scores

    def run(self):
        """
        Running the algorithm
        :return:
        """

        try:

            print("Start pop algorithm")
            if self.curr_step_id == 0:
                # Evaluation of population at initialization and recording data
                self.evaluate_pop_record_step_data(n_replaced=0, record_step_data=True)
            else:
                # Evaluation of population at hot start step
                self.evaluate_pop_record_step_data(n_replaced=None, record_step_data=False)

            # Running the algorithm while the stop criterion is not reached
            while not self.stop_criterion_strategy.time_to_stop(self.output_folder_path):

                print("new step")

                if self.curr_step_id % self.print_n_steps == 0:
                    print("step : " + str(self.curr_step_id))

                # Selecting the individuals to be replaced
                curr_to_be_replaced_indices = self.select_to_be_replaced()

                # Selecting the individuals to be mutated in priority order
                to_be_mutated_indices = self.sort_to_be_mutated()

                n_mutated_tries = 0
                replaced_during_step = []

                if self.problem_type == "max":
                    print("best : " + str(self.pop[np.argmax(self.curr_total_scores)]))
                elif self.problem_type == "min":
                    print("best : " + str(self.pop[np.argmin(self.curr_total_scores)]))

                # Initialization of the object storing all generated individuals during the step
                step_gen_ind_recorder = GeneratedIndividualsRecorder(self.curr_step_id, self.evaluation_strategy)

                try:

                    # Iterating over all individuals to be replaced
                    for curr_to_be_replaced_idx in curr_to_be_replaced_indices:

                        replacement_successful = False

                        while not replacement_successful:

                            # Individual to mutate is available
                            if n_mutated_tries < len(to_be_mutated_indices):

                                curr_to_be_mutated_idx = to_be_mutated_indices[n_mutated_tries]

                                try:

                                    # Extracting the score of the individual to be replaced if defined
                                    if curr_to_be_replaced_idx < len(self.curr_total_scores):
                                        curr_to_be_replaced_total_score = self.curr_total_scores[
                                            curr_to_be_replaced_idx]
                                        curr_to_be_replaced_scores = self.curr_scores.T[curr_to_be_replaced_idx]

                                    else:
                                        curr_to_be_replaced_total_score = None
                                        curr_to_be_replaced_scores = None

                                    curr_to_be_replaced_smiles = self.pop_tabu_list[curr_to_be_replaced_idx]

                                    # Trying to perform mutation
                                    mutated_ind, mutation_desc, mutated_total_score, mutated_scores, evaluation_time = \
                                        self.mutation_strategy.mutate(
                                            individual=self.pop[curr_to_be_mutated_idx],
                                            ind_to_replace_idx=curr_to_be_replaced_idx,
                                            curr_total_score=curr_to_be_replaced_total_score,
                                            pop_tabu_list=self.pop_tabu_list,
                                            external_tabu_list=self.external_tabu_list,
                                            generated_ind_recorder=step_gen_ind_recorder)

                                    # Saving the new individual in the pop smiles list
                                    self.pop_tabu_list[curr_to_be_replaced_idx] = mutated_ind.to_aromatic_smiles()

                                    if self.record_history:

                                        if self.actions_history[curr_to_be_replaced_idx] is not None:
                                            # Recording the history of actions, score and additional values for the
                                            # replaced individual
                                            self.removed_actions_score_smi_tuple[
                                                self.actions_history[curr_to_be_replaced_idx]] = \
                                                (curr_to_be_replaced_total_score,
                                                 curr_to_be_replaced_scores,
                                                 curr_to_be_replaced_smiles)

                                        # Recording the history of actions for new individual
                                        self.actions_history[curr_to_be_replaced_idx] = \
                                            self.actions_history[curr_to_be_mutated_idx] + "|" + mutation_desc

                                    # Replacing individual
                                    self.pop[curr_to_be_replaced_idx] = mutated_ind
                                    self.n_success_mut[curr_to_be_replaced_idx] = 0
                                    self.n_fail_mut[curr_to_be_replaced_idx] = 0

                                    # Updating score
                                    self.evaluation_strategy.record_ind_score(curr_to_be_replaced_idx,
                                                                              mutated_total_score,
                                                                              mutated_scores, mutated_ind,
                                                                              evaluation_time)

                                    # Recording success
                                    replaced_during_step.append((curr_to_be_replaced_idx, mutated_ind))
                                    replacement_successful = True
                                    self.n_success_mut[curr_to_be_mutated_idx] += 1

                                except NoImproverError as err:
                                    self.n_fail_mut[curr_to_be_mutated_idx] += 1
                                    self.errors.append([self.curr_step_id, str(err)])
                                except MutationError as err:
                                    self.n_fail_mut[curr_to_be_mutated_idx] += 1
                                    self.errors.append([self.curr_step_id, str(err)])
                                except EvaluationError as err:
                                    self.n_fail_mut[curr_to_be_mutated_idx] += 1
                                    self.errors.append([self.curr_step_id, str(err)])

                                finally:
                                    n_mutated_tries += 1

                            # No more individual is available to be mutated
                            else:
                                raise NoMoreIndToMutate()

                except NoMoreIndToMutate:

                    self.errors.append([self.curr_step_id, "No more individual to be mutated"])

                    # Saving information if no individual was replaced during step
                    if len(replaced_during_step) == 0:
                        print("No replacement occurred")
                        self.errors.append([self.curr_step_id, "No replacement occured"])

                # Recording the number of replaced individuals for the step
                n_replaced = len(replaced_during_step)

                # Recording all individuals generated during the step that passed the filters (tabu and quality),
                # if necessary
                if self.record_all_generated_individuals:
                    self.all_generated_individuals_smiles.extend(np.array(step_gen_ind_recorder.smiles)[
                                                                     step_gen_ind_recorder.get_passed_filters_mask()])
                    self.all_generated_individuals_n_obj_calls.extend(np.array(step_gen_ind_recorder.objective_calls)[
                                                                    step_gen_ind_recorder.get_passed_filters_mask()])
                    self.all_generated_individuals_step.extend(np.array(step_gen_ind_recorder.get_step_vect())[
                                                                   step_gen_ind_recorder.get_passed_filters_mask()])
                    self.all_generated_individuals_obj_value.extend(np.array(step_gen_ind_recorder.total_scores)[
                                                                        step_gen_ind_recorder.get_passed_filters_mask()])
                    self.all_generated_individuals_scores = np.concatenate([
                        self.all_generated_individuals_scores,
                        step_gen_ind_recorder.get_scores_array()[step_gen_ind_recorder.get_passed_filters_mask()]
                    ])
                    self.all_generated_individuals_improver.extend(np.array(step_gen_ind_recorder.improver)[
                                                                       step_gen_ind_recorder.get_passed_filters_mask()])
                    self.all_generated_individuals_success_obj_computation.extend(
                        np.array(step_gen_ind_recorder.success_obj_computation)[
                            step_gen_ind_recorder.get_passed_filters_mask()])
                    self.all_generated_individuals_obj_computation_time.extend(np.array(
                        step_gen_ind_recorder.obj_computation_time)[step_gen_ind_recorder.get_passed_filters_mask()])

                # Informing the evaluator that the step has reached its end
                self.evaluation_strategy.end_step_population(self.pop)

                # Evaluation of new population and recording step data
                self.evaluate_pop_record_step_data(n_replaced=n_replaced, step_gen_ind_recorder=step_gen_ind_recorder)

                if self.curr_step_id % self.save_n_steps == 0:
                    self.save()

                # Updating curr step id
                self.curr_step_id += 1

            print("Stopping : stop condition reached")

        except KeyboardInterrupt:
            print("Stopping : interrupted by user")
        except MemoryError:
            print("Stopping : no memory available")
        finally:

            # Saving algorithm result
            self.save()
