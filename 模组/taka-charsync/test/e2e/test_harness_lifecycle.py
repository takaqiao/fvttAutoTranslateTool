import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from harness import Env


class HarnessLifecycleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='charsync-process-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_unknown_listener_is_not_terminated(self):
        env = Env.__new__(Env)
        env.root = self.root
        env._procs = {}
        with patch('harness._listening_pids', side_effect=[{12345}, set()]), \
                patch('harness.subprocess.run') as run:
            run.return_value.stdout = '"node.exe","12345"\n'
            with self.assertRaisesRegex(RuntimeError, '30181'):
                env._kill('main')
            run.assert_not_called()

    def test_only_owned_running_process_is_terminated(self):
        env = Env.__new__(Env)
        env.root = self.root
        proc = Mock(pid=23456)
        proc.poll.return_value = None
        env._procs = {'main': proc}
        with patch('harness._listening_pids', return_value=set()), \
                patch('harness.subprocess.run') as run:
            env._kill('main')
            run.assert_called_once_with(['taskkill', '/PID', '23456', '/T', '/F'], capture_output=True)
            proc.wait.assert_called_once_with(timeout=30)


if __name__ == '__main__':
    unittest.main()
