import json


class NProfile(object):

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        self.__attrs = {}
        self.add_attributes(**kwargs)

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
