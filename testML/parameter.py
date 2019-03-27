class Parameter:
    def __init__(self, value, error=None, unit=None, *args, **kwargs):
        self.value = value
        self.error = error
        self.unit = unit

    def __str__(self):
        if self.error is None:
            if self.unit is None:
                return f'{self.value}'
            return f'{self.value}{self.unit}'
        if self.unit is None:
            return f'{self.value}({self.error})'
        return f'{self.value}({self.error}){self.unit}'

    def __repr__(self):
        return f'Parameter(value={self.value}, error={self.error}, unit={self.unit})'


if __name__ == "__main__":
    p1 = Parameter(5777)
    p2 = Parameter(5777, 15)
    p3 = Parameter(5777, 15, 'K')
    p4 = Parameter(5777, unit='K')
    print(p1)
    print(p2)
    print(p3)
    print(p4)