import pytest
from unittest.mock import patch, mock_open
import json
from utils import load_processed_ids, save_processed_ids

def test_load_processed_ids_file_exists():
    m = mock_open(read_data='["1", "2"]')
    with patch('builtins.open', m):
        with patch('os.path.exists', return_value=True):
            ids = load_processed_ids('dummy.json')
            assert ids == {'1', '2'}

def test_load_processed_ids_file_not_found():
    with patch('os.path.exists', return_value=False):
        ids = load_processed_ids('dummy.json')
        assert ids == set()

def test_load_processed_ids_json_decode_error():
    m = mock_open(read_data='invalid json')
    with patch('builtins.open', m):
        with patch('os.path.exists', return_value=True):
            ids = load_processed_ids('dummy.json')
            assert ids == set()

def test_save_processed_ids():
    m = mock_open()
    with patch('builtins.open', m):
        save_processed_ids({'1', '2'}, 'dummy.json')
        m.assert_called_once_with('dummy.json', 'w')
        handle = m()
        # The set is converted to a list before dumping
        assert handle.write.call_args[0][0] == json.dumps(['1', '2']) or handle.write.call_args[0][0] == json.dumps(['2', '1'])
