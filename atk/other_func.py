def benchatk(model_path, data_path):
    fd = detector.Detector(os.path.join(model_path, DETECTOR_WEIGHTS_PATH))
    track_sequences_classifier = TrackSequencesClassifier(os.path.join(model_path, VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH))

    dataset = detector.UnlabeledVideoDataset(data_path)
    print('Total number of videos: {}'.format(len(dataset)))

    loader = DataLoader(dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=False, num_workers=VIDEO_NUM_WORKERS,
                        collate_fn=lambda X: X,
                        drop_last=False)

    video_name_to_score = {}

    for video_sample in loader:
        frames = video_sample[0]['frames'][:100]
        detector_frames = frames[::DETECTOR_STEP]
        video_idx = video_sample[0]['index']
        video_rel_path = dataset.content[video_idx]
        video_name = os.path.basename(video_rel_path)
        print('len', len(frames))

        if len(frames) == 0:
            video_name_to_score[video_name] = 0.5
            continue

        detections = []
        for start in range(0, len(detector_frames), DETECTOR_BATCH_SIZE):
            end = min(len(detector_frames), start + DETECTOR_BATCH_SIZE)
            detections_batch = fd.detect(detector_frames[start:end])
            for detections_per_frame in detections_batch:
                detections.append({key: value.cpu().numpy() for key, value in detections_per_frame.items()})

        tracks = detector.get_tracks(detections)
        if len(tracks) == 0:
            video_name_to_score[video_name] = 0.5
            continue
        
        sequence_track_scores = [np.array([])]
        f = 0
        l21 = 0
        for track in tracks:
            track_sequences = []
            for i, (start_idx, _) in enumerate(
                    track[:-VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH + 1:7]):
                assert start_idx >= 0 and start_idx + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH <= len(frames)
                _, bbox = track[i * 7 + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH // 2]
                track_sequences = [detector.extract_sequence(frames, start_idx, bbox, i % 2 == 0)]
                one_step, pert_size, sf = track_sequences_classifier.benchmark_atk(track_sequences, attack = args.atk)
                f += sf
                l21 += pert_size
                sequence_track_scores[0] = np.concatenate([sequence_track_scores[0], one_step])

        sequence_track_scores = np.concatenate(sequence_track_scores)
        track_probs = sequence_track_scores

        delta = track_probs - 0.5
        sign = np.sign(delta)
        pos_delta = delta > 0
        neg_delta = delta < 0
        track_probs[pos_delta] = np.clip(0.5 + sign[pos_delta] * np.power(abs(delta[pos_delta]), 0.65), 0.01, 0.99)
        track_probs[neg_delta] = np.clip(0.5 + sign[neg_delta] * np.power(abs(delta[neg_delta]), 0.65), 0.01, 0.99)
        weights = np.power(abs(delta), 1.0) + 1e-4
        video_score = float((track_probs * weights).sum() / weights.sum())

        video_name_to_score[video_name] = video_score
        print('NUM DETECTION FRAMES: {}, VIDEO SCORE: {}. {}'.format(len(detections), video_name_to_score[video_name],
                                                                     video_rel_path))
        print('fianl l21 norm {}'.format(l21))
        print('total fake frame ', f)
        
        
# this function belongs to TrackSequencesClassifier class object 
 def benchmark_atk(self, track_sequences, attack = 'white'):
        eps = 4/255
        track_sequences = [torch.stack([self.transform(image=face)['image'] for face in sequence]) for sequence in
                           track_sequences]
        track_sequences = torch.cat(track_sequences).cuda()
        
        for param in self.model.parameters():
            param.requires_grad = False
        input_var = autograd.Variable(track_sequences, requires_grad = True).cuda()
        target_var = autograd.Variable(torch.zeros(track_sequences.size(0)), requires_grad = True).cuda() # need to spoof to real, set target to real and minus the gradient later
        it = 0
        pred = np.array([1])
        last = np.array([10 ** 9])
        
        while pred.mean().item() > 0:
            
            torch.cuda.empty_cache()
            if it > 100:
                break
    
            if attack == 'bench_white':
                loss_criterion = nn.CrossEntropyLoss()
                temptotal = []
                for transform_fn in _get_transforms({"gauss_noise", "gauss_blur", "resize"}):
                    loss = 0
                    transformed_img = transform_fn(input_var)
                    logits = self.model(transformed_img).flatten()
                    loss += logits.sum()
                # pred = self.model(input_var).flatten()
                # loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target_var)
                    loss.backward()
                    temptotal.append(torch.sign(input_var.grad.detach()))
                adv = input_var.detach() - (1/255) * sum(temptotal) /3
                with torch.no_grad():
                    pred = self.model(input_var).flatten()
                #if last.mean() < pred.mean():
                #    pred = last
                #    break
                #last = pred
                # adv = input_var.detach() + (1/255) * torch.sign(input_var.grad.detach())
            elif attack == 'bench_black':
                with torch.no_grad():
                    pred = self.model(input_var).flatten()
                adv = input_var.detach() + (1/255) * self._nes_gradient_estimator(input_var, self.model)
            total_pert = adv - track_sequences
            total_pert = torch.clamp(total_pert, -eps, eps)
            input_adv = track_sequences + total_pert
            input_adv = torch.clamp(input_adv, 0, 1)
            input_var.data = input_adv.detach()
            it += 1
        img_model = torch.load(image_model_path)
        f = 0
        for i in range(len(adv)):
            t, _ = predict_image(img_model, input_var[i], 'xception')
            f += t
        l21 = torch.sum(torch.sqrt(torch.mean(torch.pow((input_var - track_sequences).unsqueeze(0), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
        track_probs = torch.sigmoid(pred).detach().cpu().numpy()
        return track_probs, l21, f
# this function belongs to TrackSequencesClassifier class object as well, but i forget what it does, i think it's useless now        
def classify(self, track_sequences):
        track_sequences = [torch.stack([self.transform(image=face)['image'] for face in sequence]) for sequence in
                           track_sequences]
        track_sequences = torch.cat(track_sequences).cuda()
        print(track_sequences.shape)
        # with torch.no_grad():
        #     # print(self.model(track_sequences))
        #     track_probs = torch.sigmoid(self.model(track_sequences)).flatten().cpu().numpy()
        # return track_probs
    
        eps = 0.006
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        modif = torch.Tensor(track_sequences.shape).fill_(0.01/255).to(device)
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=0.01)
        
        input_var = autograd.Variable(track_sequences, requires_grad = False).cuda()
        target_var = autograd.Variable(torch.zeros(track_sequences.size(0)), requires_grad = False).cuda() # need to spoof to real, set target to real and minus the gradient later
        it = 0
        maxiter = 1
        pred = np.array([1])
        last = np.array([10 ** 9])
        
        while it < self.maxiter:
            torch.cuda.empty_cache()
            print('iter', it)
                
            input_var = input_var + modifier
    
            
            loss = 0
            loss3 = 0
            logits = self.model(input_var).flatten()
            loss1 = logits.mean()
            # print(modifier.shape)
            loss2 = torch.sum(torch.sqrt(torch.mean(torch.pow(modifier.unsqueeze(0), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
            
            from lpips import LPIPS
            img_model = torch.load(image_model_path)
            target_pert = image_pert(img_model, track_sequences.unsqueeze(0), modifier.clone().detach().requires_grad_(True), 'xception')
            target_pert = target_pert.unsqueeze(0)
            lpips_distance = 0.0
            lpips_fn = LPIPS(net='alex', verbose=False).to(modifier.device)
            window_size = modifier.shape[0]
            for w in range(window_size):
                loss3 += torch.sum(torch.sqrt(torch.mean(torch.pow((modifier[w] - target_pert[0][window_size // 2]).unsqueeze(0), 2), dim=0).mean(dim=1).mean(dim=1).mean(dim=0)))

            # ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            # tv = TotalVariation().to(device)
            # mse = torch.nn.MSELoss()
            # treg = 0
            # for w in range(window_size - 1):
            #     # treg += tv(torch.abs(modifier[0][w + i] - modifier[0][w + i + 1]).unsqueeze(0))
            #     # treg += ssim(modifier[0][w + i].unsqueeze(0), modifier[0][w + i + 1].unsqueeze(0))
            #     treg += mse(modifier[0][w].unsqueeze(0), modifier[0][w + 1].unsqueeze(0))
            # # reg += -torch.log(treg / window_size)
            # reg += (treg / window_size)
                 
            
            
            loss = loss1 + loss2 + loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                pred = self.model(input_var).flatten()
                score = self.get_score(torch.sigmoid(pred).detach().cpu().numpy())
            if score < 0.02: # early abort
                break
            
            it += 1
        # check image detector performance
        f = 0
        for i in range(len(input_var)):
            t, _ = predict_image(img_model, input_var[i], model_type)
            f += t
            if t != 0:
                print('f', i, end = ' ')
        track_probs = torch.sigmoid(pred).detach().cpu().numpy()
        return track_probs, [loss2, f]
        
# for benchmark robust
# transform functions used by hussian's attack
import torchgeometry as tgm
def compress_decompress(image,factor,cuda=True):
  # :param image: tensor image
  # :param cuda: enables cuda, must be the same parameter as the model
  # using interpolate(input, size, scale_factor, mode, align_corners) to upsample
  # uses either size or scale factor

  image_size = list(image.size())
  image_size = image_size[2:]
  compressed_image = nn.functional.interpolate(image, scale_factor = factor, mode = "bilinear", align_corners = True)
  decompressed_image = nn.functional.interpolate(compressed_image, size = image_size, mode = "bilinear", align_corners = True)

  return decompressed_image



def add_gaussian_noise(image,std,cuda=True):
  # :param image: tensor image
  # :param amount: amount of noise to be added
  # :param cuda: enables cuda, must be the same parameter as the model


  new_image = image + std*torch.randn_like(image)
  new_image = torch.clamp(new_image, min=0, max=1)

  return new_image


def gaussian_blur(image,kernel_size=(11,11),sigma=(10.5, 10.5),cuda=True):
  # smooths the given tensor with a gaussian kernel by convolving it to each channel. 
  # It suports batched operation.
  # :param image: tensor image
  # :param kernel_size (Tuple[int, int]): the size of the kernel
  # :param sigma (Tuple[float, float]): the standard deviation of the kernel

  gauss = tgm.image.GaussianBlur(kernel_size, sigma)

  # blur the image
  img_blur = gauss(image)

  # convert back to numpy. Turned off because I'm returning tensor
  #image_blur = tgm.tensor_to_image(img_blur.byte())

  return img_blur


def translate_image(image, shift_x = 20, shift_y = 20, cuda=True):

  image_size = list(image.size())
  image_size = image_size[2:]

  h, w = image_size[0], image_size[1]  


  #because height indexed from zero
  points_src = torch.FloatTensor([[
    [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
  ]])

  # the destination points are the image vertexes
  points_dst = torch.FloatTensor([[
    [0 + shift_x, 0 + shift_y], [w-1+shift_x, 0 + shift_y], [w-1+shift_x, h-1 + shift_y], [0 + shift_x, h-1 + shift_y],]])

  if cuda:
    points_src = points_src.cuda()
    points_dst = points_dst.cuda()
  # compute perspective transform
  print(type(points_src), points_src)
  M = cv2.getPerspectiveTransform(points_src.cpu().detach().numpy(), points_dst.cpu().detach().numpy()) # try to use library in cv
  M = torch.from_numpy(M).float().to(device='cuda')
  # warp the original image by the found transform
  img_warp = tgm.warp_perspective(image, M, dsize=(h, w))

  return img_warp

def _get_transforms(apply_transforms = {"gauss_noise", "gauss_blur", "translation", "resize"}):
        cuda = True
        transform_list = [
            lambda x: x,
        ]

        if "gauss_noise" in apply_transforms:
            transform_list += [
                lambda x: add_gaussian_noise(x, 0.01, cuda = cuda),
            ]

        if "gauss_blur" in apply_transforms:
            transform_list += [
                lambda x: gaussian_blur(x, kernel_size = (5, 5), sigma=(5., 5.), cuda = cuda),
                lambda x: gaussian_blur(x, kernel_size = (5, 5), sigma=(10., 10.), cuda = cuda),
                lambda x: gaussian_blur(x, kernel_size = (7, 7), sigma=(5., 5.), cuda = cuda),
                lambda x: gaussian_blur(x, kernel_size = (7, 7), sigma=(10., 10.), cuda = cuda),
            ]

        if "translation" in apply_transforms:
            transform_list += [
                lambda x: translate_image(x, 10, 10, cuda = cuda),
                lambda x: translate_image(x, 10, -10, cuda = cuda),
                lambda x: translate_image(x, -10, 10, cuda = cuda),
                lambda x: translate_image(x, -10, -10, cuda = cuda),
                lambda x: translate_image(x, 20, 20, cuda = cuda),
                lambda x: translate_image(x, 20, -20, cuda = cuda),
                lambda x: translate_image(x, -20, 10, cuda = cuda),
                lambda x: translate_image(x, -20, -20, cuda = cuda),
            ]

        if "resize" in apply_transforms:
            transform_list += [
                lambda x: compress_decompress(x, 0.1, cuda = cuda),
                lambda x: compress_decompress(x, 0.2, cuda = cuda),
                lambda x: compress_decompress(x, 0.3, cuda = cuda),
            ]

        return transform_list