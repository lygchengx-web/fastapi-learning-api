import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from httpx import ASGITransport, AsyncClient
from sklearn.datasets import load_digits

from model import DigitModel
from serve import create_app
from train import train


class APITests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        folder = TemporaryDirectory(dir=Path(__file__).resolve().parent)
        cls.addClassCleanup(folder.cleanup)
        cls.model_path = Path(folder.name) / "model.json"
        cls.metrics = train(cls.model_path)

    async def test_batch_matches_loaded_model(self):
        app = create_app(self.model_path)
        images, _ = load_digits(return_X_y=True)
        inputs = images[:3].astype(int)
        expected = DigitModel.load(self.model_path).probabilities(inputs)
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                readiness = await client.get("/readyz")
                response = await client.post("/v1/predict", json={"images": inputs.tolist()})
        self.assertEqual(readiness.status_code, 200)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["model_id"], readiness.json()["model_id"])
        self.assertEqual(len(result["predictions"]), len(inputs))
        for row, prediction in zip(expected, result["predictions"]):
            self.assertEqual(prediction["digit"], int(row.argmax()))
            self.assertAlmostEqual(prediction["probability"], float(row.max()))
        self.assertGreater(self.metrics["accuracy"], 0.9)
        self.assertGreater(self.metrics["shifted_accuracy"], self.metrics["clean_only_model_shifted_accuracy"])

    async def test_bad_shape_pixel_and_batch_size_are_rejected(self):
        app = create_app(self.model_path)
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                for invalid in ([[0] * 63], [[17] * 64], [["2"] * 64], [], [[0] * 64] * 65):
                    response = await client.post("/v1/predict", json={"images": invalid})
                    self.assertEqual(response.status_code, 422)

    async def test_missing_model_fails_readiness(self):
        app = create_app(self.model_path.with_name("missing.json"))
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                self.assertEqual((await client.get("/healthz")).status_code, 200)
                self.assertEqual((await client.get("/readyz")).status_code, 503)
                response = await client.post("/v1/predict", json={"images": [[0] * 64]})
                self.assertEqual(response.status_code, 503)

    def test_invalid_artifact_is_rejected(self):
        broken = self.model_path.with_name("broken.json")
        payload = json.loads(self.model_path.read_text(encoding="utf-8"))
        payload["scale"][0] = 0
        broken.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid digit model artifact"):
            DigitModel.load(broken)


if __name__ == "__main__":
    unittest.main()
