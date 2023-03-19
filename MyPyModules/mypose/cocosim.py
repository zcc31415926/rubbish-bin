# code based on https://github.com/juxuan27/stable-diffusion/blob/v2/ldm/metrics/coco_similarity.py
import numpy as np
import copy
import time
import datetime
from xtcocotools.cocoeval import COCOeval


class COCOevalSimilarity(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='keypoints', sigmas=None, use_area=True):
        super().__init__(cocoGt, cocoDt, iouType, sigmas, use_area)
        self.params.similarityThrs = np.linspace(0.5, 0.95, 10, endpoint=True)
        self.params.similarityType = self.params.iouType

    def evaluateSimilarity(self, strategy='weightedCosineSimilarity'):
        print('[  LOG  ] running per image evaluation...')
        # add backward compatibility if useSegm is specified in params
        print(f'[  LOG  ] evaluation similarity: {strategy}')
        tic = time.time()
        self.params.imgIds = list(np.unique(self.params.imgIds))
        if self.params.useCats:
            self.params.catIds = list(np.unique(self.params.catIds))
        self.params.maxDets = sorted(self.params.maxDets)
        self._prepare()
        # loop through images, area range, max detection number
        catIds = self.params.catIds if self.params.useCats else [-1]
        self.similarities = {(imgId, catId): self.computeSimilarity(imgId, catId, strategy) \
            for imgId in self.params.imgIds for catId in catIds}
        evaluateImg = self.evaluateImgSimilarity
        maxDet = self.params.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds for areaRng in self.params.areaRng for imgId in self.params.imgIds]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('[  LOG  ] done (t={:0.2f}s).'.format(toc - tic))

    def evaluateImgSimilarity(self, imgId, catId, aRng, maxDet):
        if self.params.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in self.params.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in self.params.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None
        for g in gt:
            tmp_area = g['area'] if 'area' in g and self.use_area else g['bbox'][2] * g['bbox'][3] * 0.53
            g['_ignore'] = 1 if g['ignore'] or tmp_area < aRng[0] or tmp_area > aRng[1] else 0
        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d[self.score_key] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[: maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed similarityThrs
        similarities = self.similarities[imgId, catId][:, gtind] \
            if len(self.similarities[imgId, catId]) > 0 else self.similarities[imgId, catId]
        T = len(self.params.similarityThrs)
        G = len(gt)
        D = len(dt)
        gtm = -np.ones((T, G), dtype=np.int64)
        dtm = -np.ones((T, D), dtype=np.int64)
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if len(similarities) > 0:
            for tind, t in enumerate(self.params.similarityThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m = -1 -> unmatched)
                    similarity = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] >= 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        # since all the rest of g's are ignored as well because of the prior sorting
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if similarities[dind, gind] < similarity:
                            continue
                        # if match successful and best so far, store appropriately
                        similarity = similarities[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm < 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': imgId, 'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt], 'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm, 'gtMatches': gtm,
            'dtScores': [d[self.score_key] for d in dt],
            'gtIgnore': gtIg, 'dtIgnore': dtIg,
        }

    def computeSimilarity(self, imgId, catId, strategy='weightedCosineSimilarity'):
        assert strategy in ['weightedCosineSimilarity', 'cosineSimilarity'], \
            f'[ ERROR ] strategy {strategy} not supported'
        # dimensions here should be N*m
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d[self.score_key] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > self.params.maxDets[-1]:
            dts = dts[: self.params.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        similarity = np.zeros((len(dts), len(gts)))
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            if self.params.similarityType == 'keypoints_wholebody':
                body_gt = gt['keypoints']
                foot_gt = gt['foot_kpts']
                face_gt = gt['face_kpts']
                lefthand_gt = gt['lefthand_kpts']
                righthand_gt = gt['righthand_kpts']
                wholebody_gt = body_gt + foot_gt + face_gt + lefthand_gt + righthand_gt
                g = np.array(wholebody_gt)
            elif self.params.similarityType == 'keypoints_foot':
                g = np.array(gt['foot_kpts'])
            elif self.params.similarityType == 'keypoints_face':
                g = np.array(gt['face_kpts'])
            elif self.params.similarityType == 'keypoints_lefthand':
                g = np.array(gt['lefthand_kpts'])
            elif self.params.similarityType == 'keypoints_righthand':
                g = np.array(gt['righthand_kpts'])
            else:
                g = np.array(gt['keypoints'])
            xg = g[0 : : 3]
            yg = g[1 : : 3]
            vg = g[2 : : 3]
            if 'weighted' in strategy.lower():
                xg /= self.sigmas
                yg /= self.sigmas
            xg = xg[vg > 0]
            yg = yg[vg > 0]
            # normalize
            xg = (xg - xg.min()) / (xg.max() - xg.min())
            yg = (yg - yg.min()) / (yg.max() - yg.min())
            # l2 normalize
            xg /= np.linalg.norm(xg)
            yg /= np.linalg.norm(yg)
            g_ = np.concatenate((xg.reshape(1, -1), yg.reshape(1, -1)), 0).reshape(-1)
            for i, dt in enumerate(dts):
                if self.params.similarityType == 'keypoints_wholebody':
                    body_dt = dt['keypoints']
                    foot_dt = dt['foot_kpts']
                    face_dt = dt['face_kpts']
                    lefthand_dt = dt['lefthand_kpts']
                    righthand_dt = dt['righthand_kpts']
                    wholebody_dt = body_dt + foot_dt + face_dt + lefthand_dt + righthand_dt
                    d = np.array(wholebody_dt)
                elif self.params.similarityType == 'keypoints_foot':
                    d = np.array(dt['foot_kpts'])
                elif self.params.similarityType == 'keypoints_face':
                    d = np.array(dt['face_kpts'])
                elif self.params.similarityType == 'keypoints_lefthand':
                    d = np.array(dt['lefthand_kpts'])
                elif self.params.similarityType == 'keypoints_righthand':
                    d = np.array(dt['righthand_kpts'])
                else:
                    d = np.array(dt['keypoints'])
                xd = d[0 : : 3]
                yd = d[1 : : 3]
                # vd = d[2 : : 3]
                if 'weighted' in strategy.lower():
                    xd /= self.sigmas
                    yd /= self.sigmas
                xd = xd[vg > 0]
                yd = yd[vg > 0]
                # normalize
                xd = (xd - xd.min()) / (xd.max() - xd.min())
                yd = (yd - yd.min()) / (yd.max() - yd.min())
                # l2 normalize
                xd /= np.linalg.norm(xd)
                yd /= np.linalg.norm(yd)
                d_ = np.concatenate((xd.reshape(1, -1), yd.reshape(1, -1)), 0).reshape(-1)
                if 'cosinesimilarity' in strategy.lower():
                    similarity[i, j] = np.sum(g_ * d_) / np.linalg.norm(g_) / np.linalg.norm(d_)
                # if 'cosinedistances' in strategy.lower():
                    # cosine_similarity = np.sum(g_ * d_) / np.linalg.norm(g_) / np.linalg.norm(d_)
                    # similarity[i, j] = np.sqrt(2 * (1 - cosine_similarity))
        return similarity

    def accumulateSimilarity(self, p=None):
        print('[  LOG  ] accumulating evaluation results...')
        assert self.evalImgs is not None, '[ ERROR ] please run the evaluation function first'
        tic = time.time()
        # allows input customized parameters
        p = self.params if p is None else p
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.similarityThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        # -1 for the precision of absent categories
        precision = -np.ones((T, R, K, A, M))
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))
        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][: maxDet] for e in E])
                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]
                    dtm = np.concatenate([e['dtMatches'][:, : maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, : maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    # https://github.com/cocodataset/cocoapi/pull/332/
                    tps = np.logical_and(dtm >= 0, np.logical_not(dtIg))
                    fps = np.logical_and(dtm < 0, np.logical_not(dtIg))
                    tp_sum = np.cumsum(tps, axis=1).astype(np.float64)
                    fp_sum = np.cumsum(fps, axis=1).astype(np.float64)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R, ))
                        ss = np.zeros((R, ))
                        recall[t, k, a, m] = rc[-1] if nd else 0
                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = list(pr)
                        q = list(q)
                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]
                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        toc = time.time()
        print('[  LOG  ] done (t={:0.2f}s).'.format(toc - tic))

    def summarizeSimilarity(self):
        def _summarize(ap=1, similarityThr=None, areaRng='all', maxDets=100):
            aind = [i for i, aRng in enumerate(self.params.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(self.params.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimensions of precision: T*R*K*A*M
                s = self.eval['precision']
                # IoU
                if similarityThr is not None:
                    t = np.where(similarityThr == self.params.similarityThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimensions of recall: T*K*A*M
                s = self.eval['recall']
                if similarityThr is not None:
                    t = np.where(similarityThr == self.params.similarityThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if (s <= -1).all():
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12, ))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, similarityThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, similarityThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps_crowd():
            # Adapted from https://github.com/Jeff-sjtu/CrowdPose
            stats = np.zeros((9, ))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, similarityThr=0.5)
            stats[2] = _summarize(1, maxDets=20, similarityThr=0.75)
            stats[3] = _summarize(0, maxDets=20)
            stats[4] = _summarize(0, maxDets=20, similarityThr=0.5)
            stats[5] = _summarize(0, maxDets=20, similarityThr=0.75)
            type_result = self.get_type_result(first=0.2, second=0.8)
            stats[6] = type_result[0]
            stats[7] = type_result[1]
            stats[8] = type_result[2]
            return stats

        def _summarizeKps():
            stats = np.zeros((10, ))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, similarityThr=0.5)
            stats[2] = _summarize(1, maxDets=20, similarityThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, similarityThr=0.5)
            stats[7] = _summarize(0, maxDets=20, similarityThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('[ ERROR ]Please run accumulate() first')
        similarityType = self.params.similarityType
        if similarityType in ['segm', 'bbox']:
            summarize = _summarizeDets
        elif similarityType == 'keypoints_crowd':
            summarize = _summarizeKps_crowd
        elif 'keypoints' in similarityType:
            summarize = _summarizeKps
        self.statsSimilarity = summarize()

