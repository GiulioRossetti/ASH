import json


class NProfile(object):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        self.__attrs = {}
        self.add_attributes(**kwargs)
        self.__stats = {}

    def add_attribute(self, key: str, value: object) -> None:
        """

        :param key:
        :param value:
        :return:
        """
        self.__attrs[key] = value

    def add_attributes(self, **kwargs) -> None:
        """

        :param kwargs:
        :return:
        """
        for key, value in kwargs.items():
            self.add_attribute(key, value)

    def get_attribute(self, key: str) -> object:
        """

        :param key:
        :return:
        """
        if key in self.__attrs:
            return self.__attrs[key]
        raise ValueError(f"Attribute {key} not present in the profile.")

    def get_attributes(self) -> dict:
        """

        :return:
        """
        return self.__attrs

    def has_attribute(self, key: str):
        """

        :param key:
        :return:
        """
        return key in self.__attrs

    def add_statistic(self, attr_name: str, stat_name: str, value: float) -> None:
        """

        :param attr_name:
        :param stat_name:
        :param value:
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

        :param attr_name:
        :param stats_name:
        :return:
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

        :param attr_name:
        :param stats_name:
        :return:
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

        :param attr_name:
        :return:
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
