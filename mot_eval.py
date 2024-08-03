def motMetricsEnhancedCalculator(gtSource, tSource):
    import motmetrics as mm
    from motmetrics.mot import MOTAccumulator
    import numpy as np

    #Load the ground truth data
    gt = np.loadtxt(gtSource, delimiter=',')

    #Load the tracker output
    t = np.loadtxt(tSource, delimiter= ',')

    #Create the accumulator to update during each frame
    acc = MOTAccumulator(auto_id=True)

    #Max frame no. could differ for gt and t
    for frame in range(int(gt[:,0].max())):
        frame = +1 #Frame no. always starts from 1

        gt_dets = gt[gt[:,0] == frame, 1:6] #Select all detections from the ground truth
        t_dets = t[t[:,0]==frame, 1:6] #Select all detections from the tracker output

        C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], max_iou=0.5) #Format = gt, t

        #Call update once for a frame
        #Call them as gt_object_ids, t_object_ids, distance
        acc.update(gt_dets[:,0].astype('int').tolist(), 
                   t_dets[:,0].astype('int').tolist(), C)
    
    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                       'recall', 'precision', 'num_objects', \
                                       'mostly_tracked', 'partially_tracked', \
                                       'mostly_lost', 'num_false_positives', \
                                       'num_misses', 'num_switches', \
                                       'num_fragmentations', 'mota', 'motp' \
                                       ], \
                        name = 'acc')
    strsummary = mm.io.render_summary(
        summary, 
        namemap = {'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
               'precision': 'Prcn', 'num_objects': 'GT', \
               'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
               'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
               'num_misses': 'FN', 'num_switches' : 'IDsw', \
               'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP', \
               }
    )

    print(strsummary)

if __name__ == "__main__":
    ground_truth = 'gt/gt4.txt'
    tracking_pred = 'tracking_results4.txt'
    motMetricsEnhancedCalculator(ground_truth, tracking_pred)
