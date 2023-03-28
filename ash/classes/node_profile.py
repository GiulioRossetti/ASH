import json


class NProfile(object):
    """
    Node Profile representation. A node profile encloses attributes and statistics of a node.

    :param node_id: a node's unique identifier
    :param kwargs: attributes to be added at creation
    """

    def __init__(self, node_id: int = None, **kwargs):

        self.__attrs = {}
        self.add_attributes(**kwargs)
        self.__stats = {}
        self.node_id = node_id

    def add_attribute(self, key: str, value: object) -> None:
        """
        Sets attribute to node profile.

        :param key: attribute name
        :param value: attribute value
        :return:
        """
        self.__attrs[key] = value

    def add_attributes(self, **kwargs) -> None:
        """
        Sets attributes to node profile.

        :param kwargs: dict keyed by attribute name to be added
        :return:
        """
        for key, value in kwargs.items():
            self.add_attribute(key, value)

    def get_attribute(self, key: str) -> object:
        """
        Returns node attribute in the form of a tid : attribute_value dict. Raises ValueError if the attribute is not
        found :param key: attribute name :return:
        """
        if key in self.__attrs:
            return self.__attrs[key]
        raise ValueError(f"Attribute {key} not present in the profile.")

    def get_attributes(self) -> dict:
        """
        Returns all node attributes as key:value pairs. Values are dictionaries keyed by temporal id.

        :return: the node's dict of attributes
        """
        return self.__attrs

    def has_attribute(self, key: str) -> bool:
        """
        Checks if the Node Profile has attribute :key:

        :param key: attribute name
        :return: True if the Node Profile has the attribute, False otherwise.
        """
        return key in self.__attrs

    def add_statistic(self, attr_name: str, stat_name: str, value: float) -> None:
        """
        Adds a statistic related to a node attribute

        :param attr_name: attribute name
        :param stat_name: statistic value
        :param value: statistic value
        :return:
        """

        if attr_name not in self.__attrs:
            raise ValueError(f"{attr_name} not present in the profile")

        if attr_name in self.__stats:
            self.__stats[attr_name][stat_name] = value
        else:
            self.__stats[attr_name] = {stat_name: value}

    def get_statistic(self, attr_name: str, stats_name: str = None) -> dict:
        """
        Returns a statistic related to a node attribute, in the form of stat_name: stat_value pairs.

        :param attr_name: attribute name
        :param stats_name: statistic name
        :return: dictionary with the statistic
        """
        if attr_name not in self.__attrs:
            raise ValueError(f"{attr_name} not present in the profile")

        if attr_name not in self.__stats:
            raise ValueError(f"{attr_name} does not have any computed statistic")

        if stats_name is None:
            return self.__stats[attr_name]
        else:
            if stats_name in self.__stats[attr_name]:
                return {stats_name: self.__stats[attr_name][stats_name]}
            else:
                raise ValueError(f"{stats_name} is not computed for {attr_name}")

    def has_statistic(self, attr_name: str, stats_name: str) -> bool:
        """
        Checks if the Node Profile has statistic :stats_name: for attribute :attr_name:

        :param attr_name: attribute name
        :param stats_name: statistic name
        :return: True if the Node Profile has the statistic for the attribute, False otherwise.
        """
        if attr_name not in self.__attrs:
            raise ValueError(f"{attr_name} not present in the profile")

        if attr_name not in self.__stats:
            return False
        if stats_name not in self.__stats[attr_name]:
            return False
        return True

    def attribute_computed_statistics(self, attr_name: str) -> list:
        """
        Returns the Node Profile statistics' names computed for :attr_name:

        :param attr_name: attribute name
        :return: list of the statistics' names computed for the Node Profile
        """
        if attr_name not in self.__attrs:
            raise ValueError(f"{attr_name} not present in the profile")
        if attr_name not in self.__stats:
            raise ValueError(f"{attr_name} does not have any computed statistic")
        return list(self.__stats[attr_name].keys())

    def items(self) -> list:
        """

        :return:
        """
        return self.get_attributes().items()

    def __eq__(self, other: object) -> bool:
        """

        :param other:
        :return:
        """
        for key, value in self.__attrs.items():
            if not other.has_attribute(key):
                return False

            value2 = other.get_attribute(key)
            if value != value2:
                return False
        return True

    def __ge__(self, other: object) -> bool:
        """

        :param other:
        :return:
        """
        for key, value in self.__attrs.items():

            if not isinstance(value, str):
                if not other.has_attribute(key):
                    return False

                value2 = other.get_attribute(key)
                if value < value2:
                    return False
        return True

    def __le__(self, other: object) -> bool:
        """

        :param other:
        :return:
        """
        for key, value in self.__attrs.items():

            if not isinstance(value, str):
                if not other.has_attribute(key):
                    return False

                value2 = other.get_attribute(key)
                if value > value2:
                    return False
        return True

    def __str__(self) -> str:
        """

        :return:
        """
        return json.dumps(self.__attrs, indent=2)

    def to_dict(self) -> dict:
        """
        Returns a dict representation of the Node Profile

        :return:
        """
        descr = {"node_id": self.node_id, "attrs": self.__attrs}

        return descr
