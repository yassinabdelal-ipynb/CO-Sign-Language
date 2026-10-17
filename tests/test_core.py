"""No camera, personal data or trained weights are needed for these tests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from isef_preprocess import training_array, roi_crop, collect_eye, keep_eye, augment
from realtime_detection import Decoder, read_mapping, replay, EyeViT, load_model

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.mapping = read_mapping(ROOT/'config/language.json')

    def test_all_commands_replay(self):
        self.assertEqual(replay(ROOT/'demos/events.jsonl',Decoder(self.mapping)),list(self.mapping.values()))

    def test_three_four_and_prefix_free(self):
        keys = list(self.mapping)
        self.assertEqual(len(keys),29)
        self.assertEqual(set(len(k.split()) for k in keys),{3,4})
        self.assertFalse(any(b.startswith(a+' ') for a in keys for b in keys))

    def test_invalid_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'map.json'
            for commands in [{}, {'up down':'A'},{'up down left':'A','up down left right':'B'}]:
                p.write_text(json.dumps({'schema_version':1,'commands':commands}))
                with self.assertRaises(ValueError): read_mapping(p)

    def test_hold_not_repeated(self):
        dec=Decoder(self.mapping)
        for i in range(4): dec.update('center',.99,i*.1)
        for i in range(100): dec.update('up',.99,.4+i*.1)
        self.assertLessEqual(len(dec.sequence),1)

    def test_low_confidence_does_not_rearm(self):
        dec=Decoder(self.mapping)
        for i in range(10): self.assertIsNone(dec.update('up',.99,i*.1))
        self.assertEqual(dec.sequence,[])
        for i in range(10,20): dec.update('center',.1,i*.1)
        self.assertFalse(dec.armed)

    def test_timeout_and_stall(self):
        dec=Decoder(self.mapping)
        dec.sequence=['up']; dec.last_event=0
        dec.update('up',.99,6.)
        self.assertEqual(dec.sequence,[])
        dec.update('center',.99,6.1)
        dec.update('center',.99,7.)
        self.assertFalse(dec.armed)
        with self.assertRaises(ValueError): dec.update('up',.99,6.9)

    def test_invalid_timing(self):
        for kwargs in [{'dwell':0},{'neutral':float('nan')},{'confidence':1.1},{'timeout':.1}]:
            with self.assertRaises(ValueError): Decoder(self.mapping,**kwargs)

    def test_validation_pixels_exact(self):
        rng=np.random.default_rng(17)
        rgb=rng.integers(0,256,(150,350,3),dtype=np.uint8)
        image=Image.fromarray(rgb)
        expected=np.asarray(image.convert('L').resize((224,224),Image.Resampling.NEAREST),dtype=np.float32)/255
        actual=training_array(image)
        self.assertEqual(actual.shape,(1,1,224,224))
        self.assertEqual(actual.dtype,np.float32)
        np.testing.assert_allclose(actual[0,0],expected,atol=6e-8,rtol=0)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'eye.png'; image.convert('L').save(path)
            with Image.open(path) as saved:
                np.testing.assert_array_equal(training_array(saved),actual)

    def test_roi_bgr_conversion_and_bounds(self):
        frame=np.zeros((30,60,3),dtype=np.uint8); frame[:,:,2]=255
        crop=roi_crop(frame,(-5,-5,30,20))
        self.assertEqual(crop.size,(350,150))
        self.assertEqual(int(np.asarray(crop)[0,0]),76)
        for roi in [(100,100,2,2),(0,0,-1,2),(0,0,float('nan'),2)]:
            with self.assertRaises(ValueError): roi_crop(frame,roi)
        with self.assertRaises(ValueError): collect_eye(frame,[[1,2]])

    def test_gate_blink_exception(self):
        self.assertFalse(keep_eye('up','closed'))
        self.assertTrue(keep_eye('blink','closed'))
        self.assertTrue(keep_eye('up','open'))
        self.assertFalse(keep_eye('blink','unknown'))
        with self.assertRaises(ValueError): keep_eye('invalid','open')

    def test_noise_is_seeded(self):
        image=Image.new('L',(350,150),128)
        a=augment(image,np.random.default_rng(1)); b=augment(image,np.random.default_rng(1))
        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        self.assertFalse(np.array_equal(np.asarray(a),np.asarray(image)))

    def test_patch_order_and_model_shape(self):
        from realtime_detection import Patches
        x=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
        self.assertEqual(Patches(2)(x)[0,0].tolist(),[0,1,4,5])
        # A small architecture fixture tests the same attention/residual operations.
        model=EyeViT(image_size=14,patch_size=7,projection_dim=4,num_heads=2,
                     transformer_layers=1,mlp_head_units=(8,4)).eval()
        with torch.inference_mode(): output=model(torch.zeros(2,1,14,14))
        self.assertEqual(tuple(output.shape),(2,6)); self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(sum(p.numel() for p in EyeViT().parameters()),1879878)

    def test_checkpoint_missing_and_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'fixture.pt'
            with self.assertRaises(FileNotFoundError): load_model(path,'cpu')
            torch.save({'model_config':{},'class_names':['wrong'],'model_state_dict':{}},path)
            with self.assertRaises(ValueError): load_model(path,'cpu')

    def test_replay_error_has_line(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.jsonl'; path.write_text('{bad json}\n')
            with self.assertRaisesRegex(ValueError,'line 1'): replay(path,Decoder(self.mapping))

    def test_missing_yolo_fails_without_download(self):
        from data_collection_pipeline import YoloEyeGate
        with self.assertRaisesRegex(FileNotFoundError,'No download'): YoloEyeGate(None)

    def test_collection_keyboard_cancel_records_and_closes(self):
        import data_collection_pipeline as collection
        from unittest.mock import MagicMock
        camera=MagicMock(); camera.isOpened.return_value=True
        camera.get.return_value=30.
        camera.read.return_value=(True,np.zeros((100,200,3),dtype=np.uint8))
        locator=MagicMock(); locator.locate.return_value=None
        cv2=MagicMock(); cv2.VideoCapture.return_value=camera; cv2.waitKey.return_value=ord('q')
        with tempfile.TemporaryDirectory() as d, patch.object(collection,'EyeLocator',return_value=locator), \
             patch.object(collection.importlib.metadata,'version',return_value='test-fixture'), \
             patch.dict('sys.modules',{'cv2':cv2}), patch.object(collection.sys,'argv',[
                 'collect','--predictor',__file__,'--gate','manual','--label','up','--consent','--output',d]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(collection.main(),130)
            summary=json.loads(next(Path(d).glob('*/metadata.json')).read_text())
            self.assertEqual(summary['status'],'cancelled')
            self.assertEqual(summary['counts']['frames'],1)
        camera.release.assert_called_once(); locator.close.assert_called_once()


if __name__=='__main__': unittest.main()
