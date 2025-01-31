# This file is part of utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

import unittest
from collections import Counter

# Classes and functions to use in tests.
import lsst.utils
from lsst.utils import doImport
from lsst.utils._packaging import getPackageDir
from lsst.utils.introspection import get_caller_name, get_class_of, get_full_type_name, get_instance_of


class GetCallerNameTestCase(unittest.TestCase):
    """Test get_caller_name

    Warning: due to the different ways this can be run
    (e.g. directly or py.test), the module name can be one of two different
    things.
    """

    def test_free_function(self):
        def test_func():
            return get_caller_name(1)

        result = test_func()
        self.assertEqual(result, f"{__name__}.test_func")

    def test_instance_method(self):
        class TestClass:
            def run(self):
                return get_caller_name(1)

        tc = TestClass()
        result = tc.run()
        self.assertEqual(result, f"{__name__}.TestClass.run")

    def test_class_method(self):
        class TestClass:
            @classmethod
            def run(cls):
                return get_caller_name(1)

        tc = TestClass()
        result = tc.run()
        self.assertEqual(result, f"{__name__}.TestClass.run")

    def test_skip(self):
        def test_func(stacklevel):
            return get_caller_name(stacklevel)

        result = test_func(2)
        self.assertEqual(result, f"{__name__}.GetCallerNameTestCase.test_skip")

        result = test_func(2000000)  # use a large number to avoid details of how the test is run
        self.assertEqual(result, "")


class TestInstropection(unittest.TestCase):
    def testTypeNames(self):
        # Check types and also an object
        tests = [
            (getPackageDir, "lsst.utils.getPackageDir"),  # underscore filtered out
            (int, "int"),
            (0, "int"),
            ("", "str"),
            (doImport, "lsst.utils.doImport.doImport"),  # no underscore
            (Counter, "collections.Counter"),
            (Counter(), "collections.Counter"),
            (lsst.utils, "lsst.utils"),
        ]

        for item, typeName in tests:
            self.assertEqual(get_full_type_name(item), typeName)

    def testUnderscores(self):
        # Underscores are filtered out unless they can't be, either
        # because __init__.py did not import it or there is a clash with
        # the non-underscore version.
        for test_name in (
            "import_test.two._four.simple.Simple",
            "import_test.two._four.clash.Simple",
            "import_test.two.clash.Simple",
        ):
            test_cls = get_class_of(test_name)
            self.assertTrue(test_cls.true())
            full = get_full_type_name(test_cls)
            self.assertEqual(full, test_name)

    def testGetClassOf(self):
        tests = [(doImport, "lsst.utils.doImport"), (Counter, "collections.Counter")]

        for test in tests:
            ref_type = test[0]
            for t in test:
                c = get_class_of(t)
                self.assertIs(c, ref_type)

    def testGetInstanceOf(self):
        c = get_instance_of("collections.Counter", "abcdeab")
        self.assertIsInstance(c, Counter)
        self.assertEqual(c["a"], 2)
        with self.assertRaises(TypeError) as cm:
            get_instance_of(lsst.utils)
        self.assertIn("lsst.utils", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
