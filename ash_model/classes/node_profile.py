import json
from typing import Any, Dict, List, Optional, Tuple, Union


class NProfile:
    """
    A profile holding attributes and computed statistics for a node.
    """

    def __init__(self, node_id: Optional[int] = None, **kwargs: Any) -> None:
        """
        Initialize a new NProfile.

        :param node_id: Optional identifier for the node.
        :param kwargs: Initial attributes to set on the profile.

        """
        self.__attrs: Dict[str, Any] = {}
        self.__stats: Dict[str, Dict[str, float]] = {}
        self.node_id: Optional[int] = node_id
        self.add_attributes(**kwargs)

    def add_attribute(self, key: str, value: Any) -> None:
        """
        Add or update a single attribute in the profile.

        :param key: Name of the attribute to add.
        :param value: Value of the attribute.

        """
        self.__attrs[key] = value

    def add_attributes(self, **kwargs: Any) -> None:
        """
        Add or update multiple attributes in the profile.

        :param kwargs: Key-value pairs of attributes to add.

        """
        for key, value in kwargs.items():
            self.add_attribute(key, value)

    def get_attribute(self, key: str) -> Union[str, int, float]:
        """
        Retrieve the value of a given attribute.

        :param key: Name of the attribute to retrieve.

        :return: The attribute's value.

        :raises ValueError: If the attribute is not present.

        """
        if key in self.__attrs:
            return self.__attrs[key]  # type: ignore
        raise ValueError(f"Attribute '{key}' not present in the profile.")

    def get_attributes(self) -> Dict[str, Any]:
        """
        Retrieve all attributes in the profile.

        :return: Dictionary of attribute names to values.

        """
        return dict(self.__attrs)

    def has_attribute(self, key: str) -> bool:
        """
        Check if an attribute exists in the profile.

        :param key: Name of the attribute to check.

        :return: True if the attribute exists, False otherwise.

        """
        return key in self.__attrs

    def add_statistic(self, attr_name: str, stat_name: str, value: float) -> None:
        """
        Add or update a computed statistic for a given attribute.

        :param attr_name: The attribute to which the statistic applies.
        :param stat_name: Name of the statistic (e.g., 'mean', 'max').
        :param value: Numeric value of the statistic.

        :raises ValueError: If the attribute is not present in the profile.

        """
        if attr_name not in self.__attrs:
            raise ValueError(f"Attribute '{attr_name}' not present in the profile.")
        self.__stats.setdefault(attr_name, {})[stat_name] = value

    def get_statistic(
        self, attr_name: str, stats_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Retrieve computed statistics for a given attribute.

        :param attr_name: The attribute whose statistics to retrieve.
        :param stats_name: Specific statistic name to retrieve (optional).

        :return: Dictionary of statistic names to their values.

        :raises ValueError: If the attribute or statistic is not present.

        """
        if attr_name not in self.__attrs:
            raise ValueError(f"Attribute '{attr_name}' not present in the profile.")
        if attr_name not in self.__stats:
            raise ValueError(f"No statistics computed for attribute '{attr_name}'.")
        if stats_name is None:
            return dict(self.__stats[attr_name])
        if stats_name in self.__stats[attr_name]:
            return {stats_name: self.__stats[attr_name][stats_name]}
        raise ValueError(
            f"Statistic '{stats_name}' not computed for attribute '{attr_name}'."
        )

    def has_statistic(self, attr_name: str, stats_name: str) -> bool:
        """
        Check if a specific statistic exists for a given attribute.

        :param attr_name: The attribute to check.
        :param stats_name: The statistic name to check.

        :return: True if the statistic exists, False otherwise.

        :raises ValueError: If the attribute is not present.

        """
        if attr_name not in self.__attrs:
            raise ValueError(f"Attribute '{attr_name}' not present in the profile.")
        return stats_name in self.__stats.get(attr_name, {})

    def attribute_computed_statistics(self, attr_name: str) -> List[str]:
        """
        List all computed statistic names for a given attribute.

        :param attr_name: The attribute to query.

        :return: List of statistic names.

        :raises ValueError: If the attribute is not present or has no statistics.

        """
        if attr_name not in self.__attrs:
            raise ValueError(f"Attribute '{attr_name}' not present in the profile.")
        if attr_name not in self.__stats:
            raise ValueError(f"No statistics computed for attribute '{attr_name}'.")
        return list(self.__stats[attr_name].keys())

    def items(self) -> List[Tuple[str, Any]]:
        """
        Get all attribute key-value pairs as a list.

        :return: List of tuples (key, value).

        """
        return list(self.__attrs.items())

    def __eq__(self, other: object) -> bool:
        """
        Check equality of two profiles based on their attributes.

        :param other: Another object to compare against.

        :return: True if the other object is an NProfile with identical attributes.

        """
        if not isinstance(other, NProfile):
            return NotImplemented
        for key, value in self.__attrs.items():
            if not other.has_attribute(key) or other.get_attribute(key) != value:
                return False
        return True

    def __ge__(self, other: object) -> bool:
        """
        Compare if all numeric attributes of this profile are greater than or equal to those of another.

        :param other: Another NProfile to compare to.

        :return: True if for every non-string attribute, this value >= other's value.

        """
        if not isinstance(other, NProfile):
            return NotImplemented
        for key, value in self.__attrs.items():
            if isinstance(value, (int, float)):
                if not other.has_attribute(key) or value < other.get_attribute(key):
                    return False
        return True

    def __le__(self, other: object) -> bool:
        """
        Compare if all numeric attributes of this profile are less than or equal to those of another.

        :param other: Another NProfile to compare to.

        :return: True if for every non-string attribute, this value <= other's value.

        """
        if not isinstance(other, NProfile):
            return NotImplemented
        for key, value in self.__attrs.items():
            if isinstance(value, (int, float)):
                if not other.has_attribute(key) or value > other.get_attribute(key):
                    return False
        return True

    def __str__(self) -> str:
        """
        Return a JSON-formatted string of all attributes.

        :return: Pretty-printed JSON string of attributes.

        """
        return json.dumps(self.__attrs, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the profile to a dictionary representation.

        :return: Dictionary with 'node_id' and 'attrs' keys.

        """
        return {"node_id": self.node_id, "attrs": dict(self.__attrs)}
