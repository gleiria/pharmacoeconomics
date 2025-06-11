#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:16:46 2024

implementation of the data structure detailed in 

Jonathan E. Fieldsend and Richard M. Everson. 2014. 
Efficiently identifying Pareto solutions when objective values change. 
In Proceedings of the 2014 Annual Conference on Genetic and Evolutionary Computation (GECCO '14). 
Association for Computing Machinery, New York, NY, USA, 605–612. 
https://doi.org/10.1145/2576768.2598279

for the maintenance of the history of solutions in uncertain multi-objective optimisation, where designs
can periodically be reevaluated to update their expected performance. The data structure maintains the
best estimate of the non-dominated set based upon the assocaited expected performance of the solutions.
and is based on the Java code hosted at https://github.com/fieldsend/

Currently uses lists -- next version will incorporate refactoring with numpy arrays in a few places

@author: Jonathan Fieldsend
@version: 1.1
"""

import random
import numpy as np
from typing import List

class Solution:
    """
    Base class to containing general methods for multi-objective solutions. Shouldn't be directly instatiated as 
    has methods raising NotImplementedError which need overriding by a concrete subtype. Assumes minimisation of all
    objectives
    """
    def __init__(self):
        pass

    def get_objective_value(self, index: int) -> float:
        """Returns the index-th objective value of this Solution"""
        raise NotImplementedError

    def get_number_of_objectives(self) -> int:
        """Returns the number of objectives"""
        raise NotImplementedError

    def dominates(self, s) -> bool:
        """Returns True if this Solution dominates s (has a lower or equal value on all objectives and strictly lower on at least one), otherwise returns False"""
        s : Solution
        better = 0
        for i in range(self.get_number_of_objectives()):
            if self.get_objective_value(i) < s.get_objective_value(i):
                better += 1
            elif self.get_objective_value(i) > s.get_objective_value(i):
                return False
        return better > 0

    def get_pareto_order(self, s) -> int:
        """
        Returns the relative order of this Solution compared to the argument s. If returns -1 then 
        this dominates or is equal to s in quality. If the method returns 1 this is dominated by s. 
        If it returns 0 this Solution and s are mutually non-dominating. 
        """
        s : Solution
        any_better = False
        any_worse = False
        i = 0
        for i in range(self.get_number_of_objectives()):
            if self.get_objective_value(i) < s.get_objective_value(i):
                any_better = True
                break
            if self.get_objective_value(i) > s.get_objective_value(i):
                any_worse = True
                break
        if any_better:
            for j in range(i, self.get_number_of_objectives()):
                if self.get_objective_value(j) > s.get_objective_value(j):
                    return 0
            return -1
        elif any_worse:
            for j in range(i, self.get_number_of_objectives()):
                if self.get_objective_value(j) < s.get_objective_value(j):
                    return 0
            return 1
        return -1

    def better(self, s) -> List[bool]:
        """Returns a boolean array, whose elements are True if this Solution is better (has a lower value) 
        than the corresponding objective element as s, otherwise the element in the boolean array is False.
        """
        s : Solution
        array = []
        for i in range(self.get_number_of_objectives()):
            array.append(self.get_objective_value(i) < s.get_objective_value(i))
        return array

    def weakly_dominates(self, s) -> bool:
        """Returns True if this solution dominates or is equal quality to s, otherwise returns False"""
        s : Solution
        for i in range(self.get_number_of_objectives()):
            if self.get_objective_value(i) > s.get_objective_value(i):
                return False
        return True

    @staticmethod
    def weakly_dominates_by_vector(d: List[float], s) -> bool:
        """Returns True if the objective vector represented by d dominates or is equal quality to Solution s, otherwise returns False"""
        s : Solution
        for i in range(len(s)):
            if d[i] > s.get_objective_value(i):
                return False
        return True

    @staticmethod
    def weakly_dominates_solution(s, d: List[float]) -> bool:
        """Returns True if Solution s dominates or is equal quality to the objective vector represented by d, otherwise returns False"""
        s : Solution
        for i in range(len(s)):
            if s.get_objective_value(i) > d[i]:
                return False
        return True

    @staticmethod
    def dominates_by_vector(d: List[float], s) -> bool:
        """Returns True if the objective vector represented by d dominates Solution s, otherwise returns False"""
        s : Solution
        better = 0
        for i in range(len(s)):
            if d[i] < s.get_objective_value(i):
                better += 1
            elif d[i] > s.get_objective_value(i):
                return False
        return better > 0

    @staticmethod
    def dominates_solution(s, d: List[float]) -> bool:
        """Returns True if Solution s dominates the objective vector represented by d, otherwise returns False"""
        s : Solution
        better = 0
        for i in range(len(s)):
            if s.get_objective_value(i) < d[i]:
                better += 1
            elif s.get_objective_value(i) > d[i]:
                return False
        return better > 0

    def strictly_dominates(self, s) -> bool:
        """
        Returns true if this Solution is better on all objectives than Solution s. 
        See e.g. Knowles et al. A tutorial on the Performance Assessment of Stochastic Multiobjective Optimizers, ETH Zurich TIK-Report 214, 2006 for Strict Dominance definition.
        """
        s : Solution
        for i in range(self.get_number_of_objectives()):
            if self.get_objective_value(i) >= s.get_objective_value(i):
                return False
        return True

    def better_or_equal(self, s) -> List[bool]:
        """Returns a boolean array, whose elements are True if this Solution is better (has a lower value) or equal value to the corresponding objective element in Solution s, otherwise the element in the boolean array is False."""
        s : Solution
        array = []
        for i in range(self.get_number_of_objectives()):
            array.append(self.get_objective_value(i) <= s.get_objective_value(i))
        return array

    def better_objectives(self, s) -> List[int]:
        """Returns a list of those objective indices for which this Solution is better than Solution s"""
        s : Solution
        array = []
        for i in range(self.get_number_of_objectives()):
            if self.get_objective_value(i) < s.get_objective_value(i):
                array.append(i)
        return array

    def equal_objectives(self, s) -> List[int]:
        """Returns a list of those objective indices for which this Solution is equal to Solution s"""
        s : Solution
        array = []
        for i in range(self.get_number_of_objectives()):
            if self.get_objective_value(i) == s.get_objective_value(i):
                array.append(i)
        return array

    def worse_or_equal_objectives(self, s) -> List[int]:
        """Returns a list of those objective indices for which this Solution is worse or equal than Solution s"""
        s : Solution
        array = []
        for i in range(self.get_number_of_objectives()):
            if self.get_objective_value(i) >= s.get_objective_value(i):
                array.append(i)
        return array

    def worse_or_equal_index(self, s, element_weights: List[int]) -> int:
        """Returns a weighted value depending on which objectives this Solution is greater or equal to Solution s on, given the element_weights"""
        s : Solution
        val = 0
        for i in range(self.get_number_of_objectives()):
            if self.get_objective_value(i) >= s.get_objective_value(i):
                val += element_weights[i]
        return val

    def equal_index(self, s, element_weights: List[int]) -> int:
        """Returns a weighted value depending on which objectives this Solution is equal to Solution s on, given the element_weights"""
        s : Solution
        val = 0
        for i in range(self.get_number_of_objectives()):
            if self.get_objective_value(i) == s.get_objective_value(i):
                val += element_weights[i]
        return val

    def is_same_quality(self, s) -> bool:
        """Returns True if both this and Solution s have objective vectors with identical values, otherwise returns False"""
        s : Solution
        for i in range(self.get_number_of_objectives()):
            if self.get_objective_value(i) != s.get_objective_value(i):
                return False
        return True


class UncertainSol(Solution):
    """
    Solution maintained by the archive where the objective vector is uncertain, and can be refined via multiple additional 
    objective vector evaluations, which can be passed in
    """
    def __init__(self, objectives_to_copy: List[float], decision_vector_to_copy: List[float]):
        """Generates a new UncertainSol with the correcponding decision vector and initial objective vector"""
        self.__objectives = objectives_to_copy.copy()
        self.__decision_vector = decision_vector_to_copy.copy()
        self.__repeated_evaluations = [objectives_to_copy.copy()]
        self.__guarded_indices = []

    def sanity_check(self, history) -> bool :
        """Should always return True. Checks that this Solution weakly dominates all Solutions it guards, and recursively checks the same holds for those guarded by its guards, etc. If returns False this property has been infringed"""
        history: List[UncertainSol]
        for i in self.__guarded_indices :
            if self.weakly_dominates(history[i]) == False :
                return False
            if history[i].sanity_check(history) == False :
                return False
        return True

    def self_guarding_check(self, index: int, history) -> bool :
        """Should always return True. Checks that this Solution does not guard itself, and recursively checks the same holds for those guarded by its guards, etc. If returns False this property has been infringed"""
        history: List[UncertainSol]
        for i in self.__guarded_indices :
            if index == i :
                return False
            if history[i].self_guarding_check(i,history) == False :
                return False
        return True

    def append_to_guarded_list(self, index : int) -> None :
        """Appends index to list of solutiosn being guarded"""
        self.__guarded_indices.append(index)

    def get_decision_vector(self) -> List[float] :
        """Returns the decision vector of the Solution stored at index"""
        return self.__decision_vector

    def get_repeated_evaluations(self) -> List[List[float]]:
        """Returns the repeated evaluations of the Solution stored at index"""
        return self.__repeated_evaluations
    
    def get_number_of_repeated_evaluations(self) -> int:
        """Returns the number of repeated evaluated in this UncertainSol"""
        return len(self.__repeated_evaluations)

    def get_estimated_objective_vector(self) -> List[float]:
        """Returns the estimated objective vector of the Solution stored at index"""
        return self.__objectives

    def get_objective_value(self, index: int) -> float:
        """Returns the estimated objective value at the corresponding index"""
        return self.__objectives[index]

    def set_objective_value(self, index: int, value: float) -> None:
        """Sets the estimated obejctive value at the corresponding index"""
        self.__objectives[index] = value

    def get_number_of_objectives(self) -> int:
        """Returns the number of objectives of this UncertainSol"""
        return len(self.__objectives)

    def add_new_evaluation(self, objective_vector: List[float]) -> None:
        """Add a new reevalution (objective vector) to be stored alongside the others for Solution, and update estimated performance accordingly"""
        self.__repeated_evaluations.append(objective_vector.copy())
        self.update_performance(objective_vector)

    def update_performance(self, objective_vector: List[float]) -> None:
        """Update the expected performance given a reevaluation"""
        raise NotImplementedError

    def get_guarded_indices(self) -> List[int] :
        """Returns the indices of solutions guarded by this UncertainSol"""
        return self.__guarded_indices
    
    def set_guarded_indices(self, to_guard : List) -> None :
        """Returns the indices of solutions guarded by this UncertainSol"""
        self.__guarded_indices = to_guard
    
class MeanPerformanceSol(UncertainSol):
    """
    Uncertain solution where the approximated multi-objective performance is taken as the mean of the 
    set of objective vectors passed into the solution from repeated reevaluations
    """
    def update_performance(self, objective_vector: List[float]) -> None:
        """ Takes in a new objective vector and incrementally updates the mean (expected) performance"""
        for i in range(self.get_number_of_objectives()):
            self.set_objective_value(i, self.get_objective_value(i) + (objective_vector[i] - self.get_objective_value(i)) / self.get_number_of_repeated_evaluations())

class UncertainObjectivesArchiver:
    """ 
    Maintains the history and elite set for an uncertain/noisy multi-objective problem, where previously entered solutions
    may have further reevaluations, which changes their performance relative to other members
    """
    def __init__(self):
        """ Initialises and sets up empty history"""
        self.__history = []
        self.__elite_indices = []
        self.__random_number_generator = random.Random()
        self.__recent_dominator_index = -1

    def self_guarding_check(self) -> bool:
        """Correctness check method, should always return True as a solution should never act as a guard to itself"""
        for i in self.__elite_indices :
            if self.__history[i].self_guarding_check(i,self.__history) == False :
                return False
        return True

    def sanity_check(self) -> bool:
        """Correctness check method, should always return True as a solution should always weakly dominate anything it guards"""
        for i in self.__elite_indices :
            if self.__history[i].sanity_check(self.__history) == False :
                return False
        return True

    def weakly_dominates(self, s: UncertainSol) -> bool:
        """Returns true if the argument solution is dominated by the elite set basedon their objective vector approximations, otherwise returns False"""
        for i in self.__elite_indices:
            if self.__history[i].weakly_dominates(s):
                self.__recent_dominator_index = i
                return True
        return False

    def __remove_dominated(self, s: UncertainSol) -> List[int]:
        """ Removes and returns the sublist of indices from the elite set which relate to solutions dominated by the solution argument"""
        to_be_guarded = []
        for i in range(len(self.__elite_indices) - 1, -1, -1):
            if s.dominates(self.__history[self.__elite_indices[i]]):
                to_be_guarded.append(self.__elite_indices.pop(i))
        return to_be_guarded

    def insert_new_solution(self, objective_vector: List[float], decision_vector: List[float]) -> int:
        """
        Creates and adds the solution represented by the decision vector and single objective vector argument. Places
        in the history tracked by the archive and updates the elite set membership if necessary. Returns the index
        at which the solution is stored -- this location will remain unchanged across the lifetime of this UncertainObjectivesArchiver
        """
        s = MeanPerformanceSol(objective_vector, decision_vector)
        inserted_index = len(self.__history)
        self.__history.append(s) #// new solution is now at index len(history)-1

        # first check not weakly dominated, if so, need to put into guarded state
        if self.weakly_dominates(s):
            is_guarded = False
            # add to guarded list
            for index in self.__history[self.__recent_dominator_index].guarded_indices:
                if self.__history[index].dominates(s):
                    # solution that is guarded by the elite set dominator also dominates the new 
                    # solution, so set as the guarding dominator
                    self.__history[index].append_to_guarded_list(len(self.__history) - 1)
                    is_guarded = True # track that it has been added to the guarded list of a solution
                    break
            # if no solution immediately guarded by the dominating member of the elite set
            # dominates the new solution, so assign as guarded by the elite set dominator
            if not is_guarded:
                self.__history[self.__recent_dominator_index].guarded_indices.append(len(self.__history) - 1)
        else: # not dominated, so need to add to elite set and update its contents
            to_be_guarded = self.__remove_dominated(s)
            # add removed to guarded list
            s.guarded_indices = to_be_guarded # update guarded list with any dominated members removed from elite set
            self.__elite_indices.append(inserted_index) # put in elite set tracking list
        return inserted_index

    def update_solution(self, index: int, reevaluation: List[float]) -> None:
        """
        Updates the solution stored at the index with an additional objective vector reevaluation. Maintains correctness of
        the elite set given this change.
        """
        updated_solution = self.__history[index]
        updated_solution.add_new_evaluation(reevaluation) # add reevaluation, which will update expected performance
        self.__elite_indices.remove(index) # index is removed from the elite list if in it, as may now be dominated

        # get set of guarded and reset to empty the guarded for the (previously) elite member
        guarded = updated_solution.get_guarded_indices() # get a copy of the guarded solutions
        updated_solution.set_guarded_indices([]) # redirect to an empty list

        if not self.weakly_dominates(updated_solution): # elite set does not dominate updated solution
            dominated = False
            dominator = -1
            # even if not dominated by elite set, could be dominated by members of the guarded set, due to location change
            for i in guarded: # check if any previously guarded dominate it
                if self.__history[i].weakly_dominates(updated_solution):
                    dominator = i
                    dominated = True
                    break
            if not dominated: # not dominated by any previously guarding either, so need to (re)insert into elite set
                to_be_guarded = self.__remove_dominated(updated_solution) # first take out any elite set members who are dominated
                updated_solution.guarded_indices = to_be_guarded # assign to be guarded by the new elite member
                self.__elite_indices.append(index) # (re)insert reevaluated solution into elite set 
            else:
                self.__history[dominator].guarded_indices.append(index) # dominated by a previously guarded solution, so set it as guardian
        else:
            self.__history[self.__recent_dominator_index].guarded_indices.append(index) # dominated by an elite solution, so guard by that

        # now need to reassign the members of the previous guarded solution set for the solution before it was updated
        for c in guarded:
            guarded_solution = self.__history[c]
            not_assigned = True
            for d in guarded:
                if self.__history[d].dominates(guarded_solution):
                    self.__history[d].append_to_guarded_list(c)
                    not_assigned = False
                    break
            if not_assigned: # if a guard not identifed from the other members of the previous guarded set, look at previous dominator
                if updated_solution.weakly_dominates(guarded_solution):
                    updated_solution.guarded_indices.append(c)
                    not_assigned = False
            if not_assigned: # previous dominator no longer dominates, so look at rest of updated elite set
                # possibly could make quicker by ensuring e != element, but as the elite set grows the 
                # time spent checking for each element of elite set is likely to outweight the single extra 
                # weakdominates check saving it causes.
                for e in self.__elite_indices:  
                    if self.__history[e].weakly_dominates(guarded_solution):
                        self.__history[e].guarded_indices.append(c)
                        not_assigned = False
                        break
            if not_assigned: # not dominated due to moving of orginal guard which must have been only dominator, so add to elite set
                self.__elite_indices.append(c)

    def get_index_of_random_elite(self) -> int:
        """Return the index of an elite solution at random"""
        return self.__elite_indices[self.__random_number_generator.randint(0, len(self.__elite_indices) - 1)]

    def get_index_of_most_uncertain_elite(self) -> int:
        """Returns the index of the elite solution with the fewest reevaluations"""

        # can speed up by using an additional sorted data structure on the size such as a red black tree 
        # and change from O(n) to O(log n)
        index = self.__elite_indices[0]
        size = len(self.__history[self.__elite_indices[0]].get_repeated_evaluations())
        for i in range(1, len(self.__elite_indices)):
            if len(self.__history[self.__elite_indices[i]].get_repeated_evaluations()) < size:
                size = len(self.__history[self.__elite_indices[i]].get_repeated_evaluations())
                index = self.__elite_indices[i]
        return index

    def get_average_number_of_resamples_in_elite(self) -> float:
        """Returns the averages number of reevaluations across all elite set members"""
        total = sum(len(self.__history[i].get_repeated_evaluations()) for i in self.__elite_indices)
        return total / len(self.__elite_indices)

    def get_decision_vector_at_index(self, index: int) -> List[float]:
        """Returns the decision vector of the solution stored at the corresponding index"""
        return self.__history[index].get_decision_vector()

    def get_estimated_objective_vector_at_index(self, index: int) -> List[float]:
        """Returns the estimated objective vector of the solution stored at the corresponding index (based on the set of reevaluated stored for the solution)"""
        return self.__history[index].get_estimated_objective_vector()

    def get_repeated_evaluations_at_index(self, index: int) -> List[List[float]]:
        """Returns the repeated_evaluations (objective vectors) of the solution stored at the corresponding index"""
        return self.__history[index].get_repeated_evaluations()


    def get_index_of_random_solution(self) -> int:
        """Returns the index of a random stored solution"""
        return self.__random_number_generator.randint(0, len(self.__history) - 1)

    def get_number_of_elite(self) -> int:
        """Returns the current number of elite solutions, based upon the estimated objective vectors"""
        return len(self.__elite_indices)

    def get_number_of_reevaluations_of_most_uncertain_elite(self) -> int:
        """Returns the number of revaluations of the elite solution with the fewest reevaluations"""

        # can speed up by using an additional sorted data structure on the size such as a red black tree 
        # and change from O(n) to O(log n)
        index = self.__elite_indices[0]
        size = len(self.__history[self.__elite_indices[0]].get_repeated_evaluations())
        for i in range(1, len(self.__elite_indices)):
            if len(self.__history[self.__elite_indices[i]].get_repeated_evaluations()) < size:
                size = len(self.__history[self.__elite_indices[i]].get_repeated_evaluations())
        return size

    def get_elite_solutions(self) -> List[UncertainSol]:
        out = []
        for i in self.__elite_indices :
            out.append(self.__history[i])
        return out   
    

    # writen by Gonçalo
    def get_archive_history(self) -> List[dict]:
        """Returns the full history of solutions stored in the archive as a list of dictionaries containing decision vectors and objective vectors."""
        return [
            {
                "decision_vector": sol.get_decision_vector(),
                "objective_vector": sol.get_estimated_objective_vector(),
                "repeated_evaluations": sol.get_repeated_evaluations()
            }
            for sol in self.__history
        ]

class UncertainTester:
    """Class to illustrate the use of the archiver on a noisy three-objective problem"""
    random_number_generator = random.Random(0)

    @staticmethod
    def main():
        """ method performa a small run showing use of the UncertainObjectivesArchiver class"""
        MAX_GENS = 2000
        evals = 1
        archive = UncertainObjectivesArchiver()
        decision_vector = [UncertainTester.random_number_generator.random() for _ in range(20)]
        objectives = UncertainTester.evaluate(decision_vector)
        archive.insert_new_solution(objectives, decision_vector)
        for i in range(MAX_GENS):
            accuracy = archive.get_average_number_of_resamples_in_elite()
            # Generate new random solution
            decision_vector = [UncertainTester.random_number_generator.random() for _ in range(20)]
            objectives = UncertainTester.evaluate(decision_vector)
            archive.insert_new_solution(objectives, decision_vector)
            evals += 1
            #print(f"self guard check {archive.self_guarding_check()}")
            index = archive.get_index_of_most_uncertain_elite()
            decision_vector = archive.get_decision_vector_at_index(index)
            reevaluation = UncertainTester.evaluate(decision_vector)
            archive.update_solution(index, reevaluation)
            evals += 1
                
            #while (accuracy >= archive.get_average_number_of_resamples_in_elite()) :   
                # Reevaluate archive member
            #    index = archive.get_index_of_most_uncertain_elite()
            #    decision_vector = archive.get_decision_vector_at_index(index)
            #    reevaluation = UncertainTester.evaluate(decision_vector)
            #    archive.update_solution(index, reevaluation)
            #    evals += 1
                #print(f"while self guard check {archive.self_guarding_check()}")
            
            if i % 1 == 0:
                print(f"Iteration {i}, Evals {evals}, Number of elite solutions: {archive.get_number_of_elite()}, average number of reevals {archive.get_average_number_of_resamples_in_elite()}")
        #extra 10% refinement period
        for i in range(200):
            index = archive.get_index_of_most_uncertain_elite()
            decision_vector = archive.get_decision_vector_at_index(index)
            reevaluation = UncertainTester.evaluate(decision_vector)
            archive.update_solution(index, reevaluation)
            evals += 1
            if i % 1 == 0:
                print(f"Refinement iteration {i}, Evals {evals}, Number of elite solutions: {archive.get_number_of_elite()}, average number of reevals {archive.get_average_number_of_resamples_in_elite()}")
        
        elite_set = archive.get_elite_solutions()

    @staticmethod
    def evaluate(d) -> List[float]:
        """Returns the noisy three-objective evaluation of the decision vector argument"""
        num = 3
        output = [0] * num
        for i in range(num):
            output[i] = 0.0
        for value in d:
            if value < 0.5:
                output[0] += 1
            else:
                output[1] += 1
        output[2] += abs(output[0] - output[1])

        for i in range(num):
            output[i] = (output[i]/len(d)) + (UncertainTester.random_number_generator.random() * 0.1)-0.05
        

        return output
    
if __name__ == "__main__":
    UncertainTester.main()