import os
import sys

import pandas as pd
import pickle
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

######## GET BINARY FILES ###########

# dysglycemic to ab1
with open("transition_probabilities/binary_files/Hyperglycemia2sAB.bin","rb") as binary_dysglycemic_to_ab1:
    dysglycemic_to_ab1_model = pickle.load(binary_dysglycemic_to_ab1)

# dysglycemic to mab1
with open("transition_probabilities/binary_files/Hyperglycemia2mAB.bin","rb") as binary_dysglycemic_to_mab1:
    dysglycemic_to_mab1_model = pickle.load(binary_dysglycemic_to_mab1)

# dysglycemic to t1d
with open("transition_probabilities/binary_files/Hyperglycemia2T1D.bin","rb") as binary_dysglycemic_to_t1d:
    dysglycemic_to_t1d_model = pickle.load(binary_dysglycemic_to_t1d)


class FromDysglycemia:

    def __init__(self):
        self.name = "from_dysglycemia"
        self.completed_cycles = 0
        self.survival_probs_dysglycemic_to_ab1 = []
        self.survival_probs_dysglycemic_to_mab1 = []
        self.survival_probs_dysglycemic_to_t1d = []
    
    def setup(self, builder:Builder):
        self.population_view = builder.population.get_view(["state","previous_state","time_in_state","GRS2","fdr","age"])
        self.dysglycemic_population_view = builder.population.get_view(["state","previous_state","time_in_state","GRS2","fdr","age"], query = "state == 'dysglycemic'")

        # register rates with value_system
        self.dysglycemia_to_ab1_rate = builder.value.register_rate_producer("dysglycemia_to_ab1_rate", source= self.base_dysglycemia_to_ab1_rate)
        self.dysglycemia_to_mab1_rate = builder.value.register_rate_producer("dysglycemia_to_mab1_rate", source= self.base_dysglycemia_to_mab1_rate)
        self.dysglycemia_to_t1d_rate = builder.value.register_rate_producer("dysglycemia_to_t1d_rate", source= self.base_dysglycemia_to_t1d_rate)

        #randomness
        self.randomness = builder.randomness.get_stream("from_dysglycemic")

        #resgister determine_from_dysglycemia in event system
        builder.event.register_listener("time_step", self.determine_from_dysglycemia)

    def base_dysglycemia_to_ab1_rate(self, index:pd.Index) -> pd.Series:
        dysglycemic_population_df = self.dysglycemic_population_view.get(index)
        survival_prob = dysglycemic_to_ab1_model.predict_survival_function(dysglycemic_population_df, times=1, conditional_after=dysglycemic_population_df['time_in_state'])
        # store probs for each cycle
        self.survival_probs_dysglycemic_to_ab1.append(survival_prob)
        rate = 1 - survival_prob.iloc[0]
        return rate

    def base_dysglycemia_to_mab1_rate(self, index:pd.Index) -> pd.Series:
        dysglycemic_population_df = self.dysglycemic_population_view.get(index)
        survival_prob = dysglycemic_to_mab1_model.predict_survival_function(dysglycemic_population_df, times=1, conditional_after=dysglycemic_population_df['time_in_state'])
        # store probs for each cycle
        self.survival_probs_dysglycemic_to_mab1.append(survival_prob)
        rate = 1 - survival_prob.iloc[0]
        return rate

    def base_dysglycemia_to_t1d_rate(self, index:pd.Index) -> pd.Series:
        dysglycemic_population_df = self.dysglycemic_population_view.get(index)
        survival_prob = dysglycemic_to_t1d_model.predict_survival_function(dysglycemic_population_df, times=1, conditional_after=dysglycemic_population_df['time_in_state'])
        # store probs for each cycle
        self.survival_probs_dysglycemic_to_t1d.append(survival_prob)
        rate = 1 - survival_prob.iloc[0]
        return rate
    

    def _get_dysglycemic_population(self, full_population):
        """Get the dysglycemic population index from the full population."""
        dysglycemic_population = full_population[full_population["state"] == "dysglycemic"]
        dysglycemic_population_index = dysglycemic_population.index
        return dysglycemic_population_index
    
    def _compute_future_state(self, effective_dysglycemia_to_ab1_rate, effective_dysglycemia_to_mab1_rate, effective_dysglycemia_to_t1d_rate, dysglycemic_population_index):
        merged_df = pd.concat([effective_dysglycemia_to_ab1_rate.rename("dysglycemia_to_ab1_rate"), effective_dysglycemia_to_mab1_rate.rename("dysglycemia_to_mab1_rate"), effective_dysglycemia_to_t1d_rate.rename("dysglycemia_to_t1d_rate")], axis = 1)
        #get draw
        draw = self.randomness.get_draw(dysglycemic_population_index)
        draw, merged_df = draw.align(merged_df)

        #merge draw onto merged_df
        merged_df = pd.concat([merged_df, draw.rename("draw")], axis = 1)
        # add 'future_state' column
        merged_df["future_state"] = ""

        #logical vectors and populate 'future_state' column
        merged_df.loc[merged_df["draw"] < merged_df["dysglycemia_to_ab1_rate"], "future_state"] = "Ab1"
        merged_df.loc[(merged_df["draw"] >= merged_df["dysglycemia_to_ab1_rate"]) & (merged_df["draw"] < (merged_df["dysglycemia_to_ab1_rate"] + merged_df["dysglycemia_to_mab1_rate"])), "future_state"] = "mAb1"
        merged_df.loc[(merged_df["draw"] >= merged_df["dysglycemia_to_ab1_rate"] + merged_df["dysglycemia_to_mab1_rate"]) & (merged_df["draw"] < (merged_df["dysglycemia_to_ab1_rate"] + merged_df["dysglycemia_to_mab1_rate"] + merged_df["dysglycemia_to_t1d_rate"])), "future_state"] = "type1_diabetes"
        
        #get back to vectors so that state_table can be updated
        ab1_series = merged_df.loc[merged_df["future_state"]=="Ab1","future_state"]
        mab1_series = merged_df.loc[merged_df["future_state"]=="mAb1","future_state"]
        t1d_series = merged_df.loc[merged_df["future_state"]=="type1_diabetes","future_state"]
        t1d_series_index = t1d_series.index

        return ab1_series, mab1_series, t1d_series, 

        
    
    def determine_from_dysglycemia(self, event:Event):
        self.completed_cycles += 1

        if self.completed_cycles < 2:
            return
        
        full_population = self.population_view.get(event.index)
        dysglycemic_index = self._get_dysglycemic_population(full_population)

  
        #effective_rates
        effective_dysglycemia_to_ab1_rate = self.dysglycemia_to_ab1_rate(dysglycemic_index)
        effective_dysglycemia_to_mab1_rate = self.dysglycemia_to_mab1_rate(dysglycemic_index)
        effective_dysglycemia_to_t1d_rate = self.dysglycemia_to_t1d_rate(dysglycemic_index)

        ab1_series, mab1_series, t1d_series = self._compute_future_state(
            effective_dysglycemia_to_ab1_rate, 
            effective_dysglycemia_to_mab1_rate, 
            effective_dysglycemia_to_t1d_rate, 
            dysglycemic_index
        )
    
        # #use series to update state_table
        self.population_view.update(ab1_series.rename("state"))
        self.population_view.update(mab1_series.rename("state"))
        self.population_view.update(t1d_series.rename("state"))



            

            

            
           



        
