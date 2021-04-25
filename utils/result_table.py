import numpy as np
from collections import OrderedDict

class ResultTable:
    """

    Class to save and show result neatly.
    First column is always 'NAME' column.

    """
    def __init__(self, table_name='table', header=None, splitter='||', int_formatter='%3d', float_formatter='%.4f'):
        """
        Initialize table setting.

        :param list header: list of string, table headers.
        :param str splitter:
        :param str int_formatter:
        :param str float_formatter:
        """
        self.table_name = table_name
        self.header = header
        if self.header is not None:
            self.set_headers(self.header)
        self.num_rows = 0
        self.splitter = splitter
        self.int_formatter = int_formatter
        self.float_formatter = float_formatter

    def set_headers(self, header):
        """
        Set table headers as given and clear all data.

        :param list header: list of header strings
        :return: None
        """
        self.header = ['NAME'] + header
        self.data = OrderedDict([(h, []) for h in self.header])
        self.max_len = OrderedDict([(h, len(h)) for h in self.header])
        # {h: len(h) for h in self.header}

    def add_row(self, row_name, row_dict):
        """
        Add new row into the table.

        :param str row_name: name of the row, which will be the first column
        :param dict row_dict: dictionary containing column name as a key and column value as value.
        :return: None
        """

        # If header is not defined, fetch from input dict
        if self.header is None:
            self.set_headers(list(row_dict.keys()))

        # If input dict has new column, make one
        for key in row_dict:
            if key not in self.data:
                self.data[key] = ['-'] * self.num_rows
                self.header.append(key)

        for h in self.header:
            if h == 'NAME':
                self.data['NAME'].append(row_name)
                self.max_len[h] = max(self.max_len['NAME'], len(row_name))
            else:
                # If input dict doesn't have values for table header, make empty value.
                if h not in row_dict:
                    row_dict[h] = '-'

                # convert input dict to string
                d = row_dict[h]

                if isinstance(d, (int, np.integer)):
                    d_str = self.int_formatter % d
                elif isinstance(d, (float, np.float)):
                    d_str = self.float_formatter % d
                elif isinstance(d, str):
                    d_str = d
                else:
                    print('Table add row WARNING: Type %s converted to string' % type(d))
                    d_str = str(d)
                    # raise NotImplementedError('Type %s not implemented.' % type(d))

                self.data[h].append(d_str)
                self.max_len[h] = max(self.max_len[h], len(d_str))
        self.num_rows += 1

    def row_to_line(self, row_values):
        """
        Convert a row into string form

        :param list row_values: list of row values as string
        :return: string form of a row
        """
        value_str = []
        for i, header in enumerate(self.header):
            max_length = self.max_len[header]
            length = len(row_values[i])
            diff = max_length - length

            # Center align
            # left_space = diff // 2
            # right_space = diff - left_space
            # s = ' ' * left_space + row_values[i] + ' ' * right_space

            # Left align
            s = row_values[i] + ' ' * diff
            value_str.append(s)

        # for i, max_length in enumerate(self.max_len.values()):
        #     length = len(row_values[i])
        #     diff = max_length - length
        #
        #     # Center align
        #     # left_space = diff // 2
        #     # right_space = diff - left_space
        #     # s = ' ' * left_space + row_values[i] + ' ' * right_space
        #
        #     # Left align
        #     s = row_values[i] + ' ' * diff
        #     value_str.append(s)

        return self.splitter + ' ' + (' %s ' % self.splitter).join(value_str) + ' ' + self.splitter

    def to_string(self):
        """
        Convert a table into string form

        :return: string form of the table
        """
        size_per_col = {h: self.max_len[h] + 2 + len(self.splitter) for h in self.header}
        line_len = sum([size_per_col[c] for c in size_per_col]) + len(self.splitter)
        table_str = '\n'

        # TABLE NAME
        table_str += self.table_name + '\n'

        # HEADER
        line = self.row_to_line(self.header)
        table_str += '=' * line_len + '\n'
        table_str += line + '\n'
        table_str += self.splitter + '-' * (line_len - len(self.splitter) * 2) + self.splitter + '\n'

        # DATA
        for row_values in zip(*self.data.values()):
            line = self.row_to_line(row_values)
            table_str += line + '\n'
        table_str += '=' * line_len + '\n'
        return table_str

    def show(self):
        print(self.to_string())

    @property
    def shape(self):
        return (self.num_rows, self.num_cols)

    @property
    def num_cols(self):
        return len(self.header)