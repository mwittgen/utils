# This file is part of utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

"""Simple unit test for Task logging.
"""

import logging
import unittest

from lsst.utils.logging import getLogger


class TestLogging(unittest.TestCase):

    def testLogLevels(self):
        """Check that the new log levels look reasonable."""

        root = getLogger()

        self.assertEqual(root.DEBUG, logging.DEBUG)
        self.assertGreater(root.VERBOSE, logging.DEBUG)
        self.assertLess(root.VERBOSE, logging.INFO)
        self.assertLess(root.TRACE, logging.DEBUG)

    def testLogCommands(self):
        """Check that all the log commands work."""

        root = getLogger()

        with self.assertLogs(level=root.TRACE) as cm:
            root.trace("Trace")
            root.debug("Debug")
            root.verbose("Verbose")
            root.info("Info")
            root.warning("Warning")
            root.fatal("Fatal")
            root.critical("Critical")
            root.error("Error")

        self.assertEqual(len(cm.records), 8)

        # Check that each record has an explicit level name rather than
        # "Level N" and comes from this file (and not the logging.py).
        for record in cm.records:
            self.assertRegex(record.levelname, "^[A-Z]+$")
            self.assertEqual(record.filename, "test_logging.py")

        with self.assertLogs(level=root.DEBUG) as cm:
            # Should only issue the INFO message.
            with root.temporary_log_level(root.INFO):
                root.info("Info")
                root.debug("Debug")
        self.assertEqual(len(cm.records), 1)

        child = root.getChild("child")
        self.assertEqual(child.getEffectiveLevel(), root.getEffectiveLevel())
        child.setLevel(root.DEBUG)
        self.assertNotEqual(child.getEffectiveLevel(), root.getEffectiveLevel())


if __name__ == "__main__":
    unittest.main()