from typing import List, Tuple, Dict, Any


class NaiveBayesClassifier():
    """ 
    Implements a Naive Bayes Classifier for categorical data. 
    Expects a dictionary with string keys and string values. 
    Counts everything and creates propabilities from there. 
    For our example data this looks like this:

    {'stable': {
        'inside_table': {False: 0.0, True: 1.0},
        'distance_robot': {'very_close': 0.12941176470588237,'reachable': 0.8705882352941177,'far': 0.0},
        'distance_other_objects': {'very_close': 0.0, 'reachable': 0.792156862745098, 'far': 0.20784313725490197},
        'distance_edge': {'far': 0.43137254901960786, 'reachable': 0.5686274509803921, 'very_close': 0.0}
        },
    'unstable': {
        'inside_table': {False: 0.37991967871485943, True: 0.6200803212851406},
        'distance_robot': {'very_close': 0.09799196787148594, 'reachable': 0.30522088353413657, 'far': 0.5967871485943775},
        'distance_other_objects': {'very_close': 0.3598393574297189, 'reachable': 0.3686746987951807, 'far': 0.2714859437751004},
        'distance_edge': {'far': 0.03614457831325301,'reachable': 0.2779116465863454,'very_close': 0.6859437751004016}
        }
    }
    """

    def __init__(self):
        pass

    def _analyse_features(self, list_features: Dict) -> Dict:
        """Returns dictionary containing all existing values for each feature """

        values_per_feature = {}

        for features in list_features:
            for feature, value in features.items():
                if feature not in values_per_feature:
                    values_per_feature[feature] = set()
                values_per_feature[feature].add(value)

        return {feature: [value for value in values] for feature, values in values_per_feature.items()}

    def _create_counting_dict(self, possible_values_per_feature: Dict) -> Dict:
        """Returns dict with entries for all possible labels, features and values with all zero
        Slightly illegal nesting happens here
        """
        counting_dict = {}

        for label in self.labels:
            counting_dict[label] = {feature: {value: 0 for value in values}
                                    for feature, values in possible_values_per_feature.items()}

        return counting_dict

    def _analyse_lables(self, lables: List[Any]) -> List[Any]:
        possible_labels = set()

        for label in lables:
            if label not in possible_labels:
                possible_labels.add(label)

        return [label for label in possible_labels]

    def fit(self, list_features: List[Dict[Any, Any]], lables: List[Any]) -> None:
        """Implements the model fitting part of Naive Bayes Classifier for categorical data. 

        Args:
            list_features (List[Dict[str, Any]]): List of rows with dictionary's containing categorical data! So even if numbers are given, the classifier only counts discrete values
            lables (List[Any]): List containing a label for each row of list_features. Has to be the same length as the features. 
        """
        assert len(list_features) == len(lables), "We need the same count of lables and features"

        possible_values_per_feature = self._analyse_features(list_features)
        self.labels = self._analyse_lables(lables)

        
        # this creates dictionary based data structure for keeping the counts and propability of each feature and value combination
        value_feature_count = self._create_counting_dict(possible_values_per_feature)
        propabilities = self._create_counting_dict(possible_values_per_feature)

        for label in self.labels:
            value_feature_count[label]["total"] = 0

        for features, label in zip(list_features, lables):
            for feature, value in features.items():
                value_feature_count[label][feature][value] += 1

            value_feature_count[label]["total"] += 1

        for label in self.labels:
            for feature, counts in value_feature_count[label].items():
                if "total" == feature:
                    continue
                for value, count in counts.items():
                    propabilities[label][feature][value] = 1.0 * \
                        count / value_feature_count[label]["total"]

        self.propabilities = propabilities
        self.total_count = len(list_features)

        self.count_per_label = {}
        self.propability_per_label = {}

        for label in self.labels:
            self.count_per_label[label] = value_feature_count[label]["total"]
            self.propability_per_label[label] = 1.0 * self.count_per_label[label] / self.total_count

        print(
            f"Finished fitting model with {len(lables)} rows train data, {len(propabilities[self.labels[0]])} features and {len(self.labels)} labels")

    def predict(self, features: Dict) -> bool:
        feature_propabilities = {label: 1 * self.propability_per_label[label] for label in self.labels}
        print(feature_propabilities)
        
        for feature, value in features.items():
            for label in self.labels:
                print(f"Label: {label}, Feature: {feature}, Value: {value}")
                feature_propabilities[label] *= self.propabilities[label][feature][value]

        normalizing_factor = sum([feature_propabilities[label] for label in self.labels])
        print(normalizing_factor)

        results = [(label, normalizing_factor * feature_propabilities[label]) for label in self.labels]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[0][0]

    def predict_all(self, list_features: List[dict]) -> List[bool]:
        return [self.predict(features) for features in list_features]

    def predict_with_confidence(self, features: Dict) -> bool:
        feature_propabilities = {label: 1 * self.propability_per_label[label] for label in self.labels}

        for feature, value in features.items():
            for label in self.labels:
                feature_propabilities[label] *= self.propabilities[label][feature][value]

        normalizing_factor = sum([feature_propabilities[label] for label in self.labels])

        results = [(label, normalizing_factor * feature_propabilities[label]) for label in self.labels]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[0][0], results[0][1] - results[1][1]

    def predict_all_with_confidence(self, list_features: List[dict]) -> List[bool]:
        return [self.predict_with_confidence(features) for features in list_features]
