"""Run: python -m unittest discover -s tests -p test_director_protocol.py (no provider calls)."""
import importlib.util
import json
import sqlite3
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

spec = importlib.util.spec_from_file_location("runway_director_test", Path(__file__).parents[1] / "functions" / "runway_inline.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DirectorProtocolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        users = ModuleType("open_webui.models.users")
        users.Users = SimpleNamespace(get_user_by_id=AsyncMock(side_effect=lambda uid: SimpleNamespace(id=uid, role="user") if uid else None))
        env = ModuleType("open_webui.env")
        env.DATA_DIR = self.directory.name
        self.modules = patch.dict(sys.modules, {"open_webui.models.users": users, "open_webui.env": env})
        self.modules.start()
        self.pipe = module.Pipe()
        self.pipe.valves.RUNWAY_API_KEY = "fixture-key"
        self.pipe.valves.ENABLE_STUDIO_API = True
        self.task_id = str(uuid4())
        self.file_id = str(uuid4())
        self.job_id = str(uuid4())
        self.pipe._api = AsyncMock(return_value={"id": self.task_id})
        self.request = SimpleNamespace()

    def tearDown(self):
        self.modules.stop()
        self.directory.cleanup()

    async def call(self, operation, owner="alice", **extra):
        return json.loads(await self.pipe._studio({"version": 1, "operation": operation, "jobId": self.job_id, **extra}, {"id": owner}, self.request))

    async def submit(self, **extra):
        return await self.call("submit", confirmed=True, prompt="A train arrives", duration=5, ratio="16:9", **extra)

    async def test_capabilities_do_not_generate_and_chat_confirmation_is_preserved(self):
        result = await self.call("capabilities")
        self.assertTrue(result["enabled"])
        self.assertFalse(result["lastFrame"])
        self.assertTrue(self.pipe.valves.REQUIRE_CONFIRMATION)
        self.pipe._api.assert_not_awaited()

    async def test_requires_explicit_opt_in_and_consent(self):
        self.pipe.valves.ENABLE_STUDIO_API = False
        self.assertIn("error", await self.submit())
        self.pipe.valves.ENABLE_STUDIO_API = True
        self.assertIn("error", await self.call("submit", prompt="Test", duration=5, ratio="16:9"))
        self.pipe._api.assert_not_awaited()

    async def test_duplicate_submit_does_not_create_another_paid_task(self):
        self.assertEqual((await self.submit())["state"], "running")
        self.assertEqual((await self.submit())["state"], "running")
        self.assertEqual(self.pipe._api.await_count, 1)

    async def test_status_is_owner_scoped_and_imports_once(self):
        await self.submit()
        self.assertIn("error", await self.call("status", owner="bob"))
        self.pipe._api.return_value = {"status": "SUCCEEDED", "output": ["https://cdn.example.test/video.mp4"]}
        self.pipe._download = AsyncMock(return_value=b"test-video")
        self.pipe._save = AsyncMock(return_value=(f"/api/v1/files/{self.file_id}/content", "video.mp4"))
        self.assertEqual((await self.call("status"))["fileIds"], [self.file_id])
        self.assertEqual((await self.call("status"))["fileIds"], [self.file_id])
        self.pipe._save.assert_awaited_once()

    async def test_lost_submission_response_is_not_replayed(self):
        self.pipe._api.side_effect = RuntimeError("lost response")
        self.assertIn("error", await self.submit())
        self.assertEqual((await self.submit())["state"], "submission-unknown")
        self.assertEqual(self.pipe._api.await_count, 1)

    async def test_rejects_unsupported_references_without_generating(self):
        self.assertIn("error", await self.submit(lastFrameFileId=str(uuid4())))
        self.pipe._api.assert_not_awaited()

    async def test_failed_import_retries_download_not_generation(self):
        await self.submit()
        self.pipe._api.return_value = {"status": "SUCCEEDED", "output": ["https://cdn.example.test/video.mp4"]}
        self.pipe._download = AsyncMock(side_effect=[RuntimeError("network"), b"video"])
        self.pipe._save = AsyncMock(return_value=(f"/api/v1/files/{self.file_id}/content", "video.mp4"))
        self.assertIn("error", await self.call("status"))
        self.assertEqual((await self.call("status"))["state"], "succeeded")
        self.assertEqual(sum(call.args[1] == "POST" for call in self.pipe._api.await_args_list), 1)

    async def test_moderation_failure_is_persisted_owner_scoped_and_not_retried(self):
        submitted = await self.submit()
        self.assertEqual(submitted["providerTaskId"], self.task_id)
        self.pipe._api.return_value = {"status": "FAILED",
            "failureCode": "INPUT_PREPROCESSING.SAFETY.THIRD_PARTY",
            "failure": "Private provider text https://example.test/?token=secret"}
        result = await self.call("status")
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["failureCode"], "INPUT_PREPROCESSING.SAFETY.THIRD_PARTY")
        self.assertEqual(result["providerTaskId"], self.task_id)
        self.assertNotIn("secret", json.dumps(result))
        self.assertIn("error", await self.call("status", owner="bob"))
        self.assertEqual(await self.call("status"), result)
        self.assertEqual(await self.submit(), result)
        self.assertEqual(self.pipe._api.await_count, 2)  # One POST, one GET.

    async def test_malformed_failure_metadata_is_not_returned(self):
        await self.submit()
        for code in [None, "https://example.test/?key=secret", "A" * 129, {"secret": "value"}]:
            with self.subTest(code=code):
                self.assertIsNone(self.pipe._studio_failure_code(code))
        self.pipe._api.return_value = {"status": "FAILED", "failureCode": "Bearer secret"}
        result = await self.call("status")
        self.assertEqual(result["state"], "failed")
        self.assertNotIn("failureCode", result)

    async def test_migration_preserves_existing_job_and_does_not_resubmit(self):
        with sqlite3.connect(Path(self.directory.name) / "studio-runway-jobs.sqlite3") as db:
            db.execute("""CREATE TABLE job (owner TEXT NOT NULL, id TEXT NOT NULL,
                input_hash TEXT NOT NULL, state TEXT NOT NULL, task_id TEXT,
                options TEXT NOT NULL, files TEXT, import_until REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(owner,id))""")
            db.execute("INSERT INTO job(owner,id,input_hash,state,task_id,options) VALUES (?,?,?,?,?,?)",
                ("alice", self.job_id, "existing-hash", "failed", self.task_id, "{}"))
        db.close()
        result = await self.call("status")
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["providerTaskId"], self.task_id)
        self.assertNotIn("failureCode", result)
        self.assertEqual(await self.call("status"), result)
        self.pipe._api.assert_not_awaited()

    async def test_provider_canceled_spelling_is_terminal(self):
        await self.submit()
        self.pipe._api.return_value = {"status": "CANCELED"}
        self.assertEqual((await self.call("status"))["state"], "cancelled")
        self.assertEqual((await self.call("status"))["state"], "cancelled")
        self.assertEqual(self.pipe._api.await_count, 2)


if __name__ == "__main__":
    unittest.main()
