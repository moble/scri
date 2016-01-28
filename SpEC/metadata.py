# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import


def read_metadata(metadata_filename):
    """Read metadata file into python dictionary

    Given a standard metadata.txt file, this function turns it into a script, executes that script in a local
    namespace, and then returns the dictionary of that namespace.

    >>> import scri.SpEC as SpEC
    >>> metadata = SpEC.read_metadata('samples/metadata.txt')
    >>> metadata['relaxed_mass1']
    0.500229600569

    """
    # Turn the metadata.txt file into a script that can be executed
    metadata = convert_metadata_to_script(metadata_filename)

    # Now, execute the script in its own global and local namespaces
    g, l = {}, {}
    exec(metadata, g, l)

    #  Only return the local part of that namespace, which contains these new variable
    return l


def read_metadata_into_object(metadata_filename):
    """Read metadata file into python object

    This acts just like `read_metadata`, but returns an object containing all of the metadata fields as attributes

    >>> import scri.SpEC as SpEC
    >>> metadata = SpEC.read_metadata_into_object('samples/metadata.txt')
    >>> metadata.relaxed_mass1
    0.500229600569

    """
    # Turn the metadata.txt file into a dictionary
    l = read_metadata(metadata_filename)

    # Create an object containing all of the metadata fields as attributes
    class Metadata(object):
        def __init__(self, locals_dict):
            self.__dict__.update(locals_dict)
            self.all = self.__dict__

        def __repr__(self):
            return '{' + ',\n '.join(["'{0}': {1}".format(key, val) for key, val in self.__dict__.items()]) + ',}'

        def __str__(self):
            string = ("Simulation name: {0}\n"
                      + "Alternative names: {1}\n"
                      + "Masses: {2}, {3}\n"
                      + "Spins: {4},\n       {5}\n")
            return string.format(self.simulation_name, self.alternative_names,
                                 self.relaxed_mass1, self.relaxed_mass2,
                                 self.relaxed_spin1, self.relaxed_spin2)

        def _repr_html_(self):
            string = ("Simulation name: {0}<br/>\n"
                      + "Alternative names: {1}<br/>\n"
                      + "Masses: {2}, {3}<br/>\n"
                      + "Spins:<br/>\n&nbsp;&nbsp;&nbsp;&nbsp;{4},<br/>\n&nbsp;&nbsp;&nbsp;&nbsp;{5}\n")
            return string.format(self.simulation_name, self.alternative_names,
                                 self.relaxed_mass1, self.relaxed_mass2,
                                 self.relaxed_spin1, self.relaxed_spin2)

    metadata = Metadata(l)

    return metadata


def convert_metadata_to_script(metadata_filename):
    """Convert metadata markup into executable python

    N.B.: This function is intended primarily for use from the `read_metadata` and `read_metadata_into_object`
    functions, which are probably more useful in general.  It is included here just in case you have some other use
    for the resulting script.

    A standard metadata.txt file is close to being an executable python script that just defines a bunch of
    constants.  The three problems with the metadata.txt format are:

      1) variable names contain dashes, which is the subtraction operator in python,
      2) strings are not enclosed in quotes, and
      3) lists are not enclosed in brackets

    It is easy to correct these problems.  In particular, (1) is resolved by changing dashes to underscores in the
    identifiers.  A bug in SpEC's metadata.txt files -- whereby some comment lines are missing the initial `#` -- is
    also fixed.  The resulting script could then be run on its own to define the constants in the metadata file.

    Note that this function is not very flexible when it comes to generalizing the syntax of the metadata.txt files.
    In particular, it assumes that the right-hand sides are either numbers or strings (or lists of either numbers or
    strings).  For example, I think I've seen cases where the eccentricity is given as something like "<1e-5".  Since
    python has no "less-than" type, this is converted to a string.  But generally, this does seem to work on
    metadata.txt files in the SXS waveform repository.

    """
    import re
    assignment_pattern = re.compile(r"""([-A-Za-z0-9]+)(\s*=\s*)(.*)""")
    string_pattern = re.compile(r"""[A-DF-Za-df-z<>@]""")

    metadata = ''
    with open(metadata_filename, "r") as metadata_file:
        for line in metadata_file.readlines():

            # Fix bug where some lines of dashes are missing the comment character
            if line.startswith('-'):
                line = '#' + line

            # Deal with all assignment lines (leaving comments and unrecognized lines untouched)
            match = assignment_pattern.match(line)
            if match:
                variable, assignment, quantity = match.groups()

                # Stupid choice to make variables contain dashes
                variable = variable.replace("-", "_")

                # Strip whitespace from string; places quotation marks around them; split lists and place brackets
                # around them
                if string_pattern.search(quantity):
                    quantities = [q.strip() for q in quantity.split(",")]
                    if "," in quantity:
                        quantity = "['" + "', '".join(quantities) + "']"
                    else:
                        quantity = "'" + quantities[0] + "'"

                # Place brackets around lists of strings
                else:
                    if "," in quantity:
                        quantity = "[" + quantity + "]"

                # Recombine the modified parts of this line
                line = variable + assignment + quantity + "\n"

            # Add this line to the metadata string, whether or not it's been modified
            metadata += line

    return metadata
