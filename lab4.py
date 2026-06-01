import stat
import dataclasses
from agents import EntityAgent, UncertainAgent
from model import Location, GameState, GameAction, WizardMoves, Location, \
    Crystal, Portal, Lava, GameTransitions, Observation, EmptyTile, \
    LocationCounts, LocationDistribution, Wall
import random


class MDP:
    def __init__(self, initial_state: GameState, escape_reward: float,
                 living_reward: float, death_reward: float, discount: float):
        self.game_state = initial_state
        self.living_reward = living_reward
        self.death_reward = death_reward
        self.escape_reward = escape_reward
        self.discount = discount

    def reward(self, source: GameState, target: GameState,
               action: GameAction) -> float:
        loc = target.active_entity_location
        if isinstance(target.tile_grid[loc.row][loc.col], Lava):
            return self.death_reward
        # elif target.victory:
        #     return self.escape_reward
        # Alt reward logic for part 1 testing
        elif isinstance(target.tile_grid[loc.row][loc.col], Portal):
            return self.escape_reward
        else:
            return self.living_reward

    def transition_model(self, location: Location,
                         action: GameAction) -> LocationDistribution:
        """
        Transition model of the MDP, gives conditional probability
        distribution of result location given starting location and action
        choice.
        """
        source_state = self.game_state.replace_active_entity_location(location)
        successors = GameTransitions.get_successors(source_state)
        actions = [a for a, _ in successors]
        successor_states = [state for _, state in successors]

        if action not in actions:
            return self.transition_model(location, WizardMoves.STAY)

        # Outcomes are either the desired outcome of the action, or a random
        # other action each with 50% prob.
        possible_results = LocationCounts(self.game_state.grid_size)
        for i in range(len(actions)):
            if actions[i] == action:
                for _ in range(len(actions) + 1):
                    possible_results.add_count(
                        successor_states[i].active_entity_location)
            else:
                possible_results.add_count(
                    successor_states[i].active_entity_location)

        return possible_results.normalize()

    def transition_distribution(self, source: LocationDistribution,
                                action: GameAction) -> LocationDistribution:
        """
        Given a location distribution, calculate the new distribution that is
        a result of taking the given action.
        The easiest way to do this will involve sampling.
        """

        # DONE YOUR CODE HERE
        # 1. Create a counter to accumulate results
        resultCounts = LocationCounts(self.game_state.grid_size)

        # 2. Number of samples - higher = more accurate but slower
        NUM_SAMPLES = 1000

        # 3. Draw samples from the source distribution (Monte Carlo
        # approximation)
        for i in range(NUM_SAMPLES):
            # Sample a current location from the source distribution
            currentLoc = source.sample()

            # Get the transition model for this location and action
            # This tells us where we might end up from current_loc when
            # taking action
            outcomeDistribution = self.transition_model(currentLoc, action)

            # Sample an outcome location from that distribution
            outcomeLoc = outcomeDistribution.sample()

            # Record this outcome
            resultCounts.add_count(outcomeLoc)

        # 4. Convert counts back to a probability distribution
        return resultCounts.normalize()

class LocationValues:
    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self.value_grid = [[0.0 for _ in range(mdp.game_state.grid_size[1])] for
                           _ in range(mdp.game_state.grid_size[0])]

    def value_iteration(self, k):
        for _ in range(k):
            self.value_iteration_update()

    def value_iteration_update(self):
        """
        Perform one update of value iteration based off of the provided MDP.
        """

        # 1. Create a new empty grid for the next iteration's values
        next_value_grid = [
            [0.0 for _ in range(self.mdp.game_state.grid_size[1])] for _ in
            range(self.mdp.game_state.grid_size[0])]

        # DONE: YOUR CODE HERE, CALCULATE NEXT VALUE AS A FUNCTION OF PREVIOUS
        #  VALUE

        # 2. Loop through every location in the grid
        for row in range(self.mdp.game_state.grid_size[0]):
            for col in range(self.mdp.game_state.grid_size[1]):
                loc = Location(row, col)
                tile = self.mdp.game_state.tile_grid[row][col]

                # 3. Check if this is a TERMINAL STATE (Lava or Portal)
                #    Terminal states don't need additional work, their value
                #    is the reward
                if isinstance(tile, (Lava, Portal)):
                    # Create a state at this location to get its reward
                    stateAtLoc = (
                        self.mdp.game_state.replace_active_entity_location(
                            loc))
                    next_state = stateAtLoc  # These don't move, just exist
                    next_value_grid[row][col] = self.mdp.reward(stateAtLoc,
                                                                next_state,
                                                                None)
                    continue

                # 4. Check if this is a WALL (can't be in this state)
                if isinstance(tile, Wall):
                    next_value_grid[row][col] = 0.0
                    continue

                # 5. For non-terminal, non-wall locations: try each valid action
                bestValue = float('-inf')

                # Get valid actions from this location
                source_state = (
                    self.mdp.game_state.replace_active_entity_location(
                        loc))
                successors = GameTransitions.get_successors(source_state)
                valid_actions = [action for action, _ in successors]

                # Try each action
                for action in valid_actions:
                    # Get the probability distribution of where we'll end up
                    outcomeDist = self.mdp.transition_model(loc, action)

                    # Calculate expected value of taking this action
                    expectedValue = 0.0

                    # Sum over all possible outcome locations
                    for outcome_loc in outcomeDist.locations():
                        # Probability of ending up there
                        prob = outcomeDist.probability(outcome_loc)

                        # Create states to calculate reward
                        outcomeState = (
                            self.mdp.game_state.replace_active_entity_location(
                                outcome_loc))

                        # Immediate reward for this transition
                        reward = self.mdp.reward(source_state, outcomeState,
                                                 action)

                        # Discounted future value (using current value_grid,
                        # not next_value_grid!)
                        futureValue = self.mdp.discount * \
                                      self.value_grid[outcome_loc.row][
                                          outcome_loc.col]

                        # Add weighted contribution
                        expectedValue += prob * (reward + futureValue)

                    # Track the best action
                    bestValue = max(bestValue, expectedValue)

                # Store the best value for this location
                next_value_grid[row][col] = bestValue

        # 6. Update the current value grid with the new values
        self.value_grid = next_value_grid

        return next_value_grid


class MDPAgent(UncertainAgent):
    values: LocationValues
    current_position_estimate: LocationDistribution
    current_score_estimate: int
    mdp: MDP

    def __init__(self, mdp: MDP, value_iteration_steps=100):
        self.mdp = mdp
        self.values = LocationValues(mdp)
        self.values.value_iteration(value_iteration_steps)
        self.current_position_estimate = (
            LocationDistribution.from_game_state_uniform(
                mdp.game_state))

    def observation_likelihood(self, observation: Observation,
                               loc: Location) -> float:
        portal_loc = self.mdp.game_state.get_all_tile_locations(Portal)[0]
        portal_dist = abs(loc.row - portal_loc.row) + abs(
            loc.col - portal_loc.col)

        if abs(portal_dist - observation.approximatePortalDist) > 1:
            return 0
        else:
            return 1.0 / 3.0

    def update_prior(self, action: GameAction):
        self.current_position_estimate = self.mdp.transition_distribution(
            self.current_position_estimate, action)

    def update_belief(self, observation: Observation):
        """
        Use Bayes rule to update your beliefs about the wizard location by
        updating self.current_position_estimate.
        You have prior belief of your current estimate P(Loc), and the
        observation likelihood model (P(Obs | Loc)).
        Use these to calculate the new belief.
        """

        # We need to update our belief for all possible locations. So lets
        # start by creating a new distribution
        new_estimate = LocationDistribution.from_game_state_uniform(
            self.mdp.game_state)

        # The new distribution should be set for each location
        for loc in new_estimate.locations():
            # DONE: YOUR CODE HERE
            prior = self.current_position_estimate.probability(loc)
            likelihood = self.observation_likelihood(observation, loc)
            new_estimate.update_probability(loc, prior * likelihood)

        new_estimate.renormalize()
        self.current_position_estimate = new_estimate

    def react(self, observation: Observation) -> GameAction:
        """
        Our uncertain agent only has noisy observations to guess where in the
        dungeon it is. Use the previously implemented parts to generate an
        estimate for the distribution of possible locations the wizard might
        be at, updating every turn with a new observation, and choose the
        action based off of your value iteration policy based on your estimate.
        """

        # Use Bayes Rule to update your beliefs about where you think the
        # wizard is based off of the observation
        self.update_belief(observation)

        # Part 2: Choose the best action
        # use your calculated value iteration map of location values
        # alongside your estimated location to choose the best action given
        # your uncertain state.
        # There are multiple ways to do this, but some things to consider:
        # 1. You want to select the action which will have the highest
        # expected value, given the probability distribution of the resultant
        # states.
        # 2. The expected value of some quantity is just the weighted average
        # value of that quantity in an outcome weighted by the probability of
        # that outcome over all outcomes (and can be estimated by the average
        # of a sufficiently big sample of outcomes sampled from the
        # distribution)
        # 3. You can sample locations from any distribution, and can form
        # distributions of locations from samples
        # 4. You can find the distribution of the results of an action for a
        # given specific location
        # 5. You can calculate the reward of a specific transition as a
        # result of a specific action with a specific result
        # 6. You have an estimate of the value of each result location
        # DONE: YOUR CODE HERE
        
        action = WizardMoves.RIGHT

        # When choosing an action, we must update our prior to account for
        # the new distribution as a result of the action being taken
        self.update_prior(action)
        return action
