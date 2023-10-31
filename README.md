Personal notes from ICCV23

# Table of contents

- [Intro](#intro)
   * [Conference Stats](#conference-stats)
   * [My experience](#my-experience)
   * [If you're reading this for some weird reason and you're not me](#if-youre-reading-this-for-some-weird-reason-and-youre-not-me)
   * [Main Insights](#main-insights)
   * [Paper description format](#paper-description-format)
- [Workshops](#workshops)
   * [Video workshop](#video-workshop)
   * [Continual learning workshop](#continual-learning-workshop)
   * [Quo Vadis / State of Computer Vision](#quo-vadis-state-of-computer-vision)
   * [Efficient networks](#efficient-networks)
   * [Meta GenAI](#meta-genai)
- [Keynotes](#keynotes)
- [Papers by topic](#papers-by-topic)
   * [Diffusion](#diffusion)
      + [Image editing](#image-editing)
      + [LoRA/Adapters](#loraadapters)
      + [Enforce prompt matching](#enforce-prompt-matching)
      + [Other / better guidance](#other-better-guidance)
      + [Domain adaptation](#domain-adaptation)
      + [Removing/modifying concepts in pretrained diffusion](#removingmodifying-concepts-in-pretrained-diffusion)
      + [Not just text2image](#not-just-text2image)
      + [Security risks](#security-risks)
      + [Faster inference](#faster-inference)
      + [Unsorted](#unsorted)
   * [Data](#data)
      + [Investigations](#investigations)
      + [Dataset compression to N samples](#dataset-compression-to-n-samples)
      + [Datasets](#datasets)
      + [Synthetic labels](#synthetic-labels)
   * [Multi-modality](#multi-modality)
      + [VLMs (VQA, captioning, zero-shot classification, etc)](#vlms-vqa-captioning-zero-shot-classification-etc)
      + [CLIP Training](#clip-training)
      + [Prompt tuning](#prompt-tuning)
      + [CLIP zero-shot quality improvements](#clip-zero-shot-quality-improvements)
      + [CLIP Inference](#clip-inference)
      + [CLIP Data/abilities](#clip-dataabilities)
   * [GANs](#gans)
      + [Latent space manipulations](#latent-space-manipulations)
      + [Domain adaptation](#domain-adaptation-1)
      + [Unsorted](#unsorted-1)
   * [Training improvements](#training-improvements)
      + [Training](#training)
      + [Finetuning/other task adaptation](#finetuningother-task-adaptation)
      + [Decoding](#decoding)
      + [Losses](#losses)
      + [Federated learning](#federated-learning)
   * [Architectures](#architectures)
      + [Tricks](#tricks)
      + [Attention](#attention)
      + [Modules/Layers](#moduleslayers)
      + [Downsample/upsample](#downsampleupsample)
      + [Misc architectures](#misc-architectures)
      + [Encrypted inference](#encrypted-inference)
   * [Video](#video)
      + [Video Generation](#video-generation)
      + [Video + Audio](#video-audio)
      + [Video Editing](#video-editing)
      + [Video Stylization](#video-stylization)
      + [Video Tagging](#video-tagging)
   * [Other problems](#other-problems)
      + [Vector graphics](#vector-graphics)
      + [Style transfer](#style-transfer)
      + [3D](#3d)
      + [image2image](#image2image)
      + [Inpainting](#inpainting)
      + [Face Recognition](#face-recognition)
      + [Hair simulation/editing/animation](#hair-simulationeditinganimation)
      + [Detection](#detection)
      + [Adaptation to the unknown](#adaptation-to-the-unknown)
   * [Misc](#misc)

# Intro

## Conference Stats

- 2k+ papers accepted ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/563503be-71a2-4a07-a8e5-709e80219c18)


- per category:

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/3e481635-7882-485d-99f3-0f3cdcdc3ecc)


## My experience

- I mostly looked on Diffusion/GAN-related, architectures, training tricks, a little bit few/zero-shot, segmentation/detection.

- Overall ~110+ papers are covered (more than 5% of the conference) + some workshops + sometimes you can find random ideas

- Overall conference felt much less useful than ICCV2019 which I attentded in person previously ([notes](https://github.com/asmekal/iccv2019-notes)). Maybe the amount of interesting simple ideas is somewhat more depleted? Or the things I was interested in became more narrow? Or I became ~~older and more stup~~ more experienced? On most poster sessions 1 hour was sufficient to check all useful papers (if you don't want to weight 10-15 mins/poster to speak to author). Anyway, conference is in big part for socializing, and research is already dead outdated so it was not bad.

- Some posters were missing! or at the wrong place! some others were on workshops as well as on main conference or 2 consequtive days/poster sessions in 1 place... my strategy was just walking over all the posters, problem is some posters are added late, some are removed early - so you never know you checked everything. 1 time I tried to look specifically for 1 poster, checked it's allocated place as well as all posters in general - didn't found any... after which I stopped using the schedule as guidance

## If you're reading this for some weird reason and you're not me

Recommended order is

- [Main Insights](#main-insights)
- [Workshops](#workshops) and [Keynotes](#keynotes) - relatively short sections but they can provide some high-level interesting ideas
- [Paper description format](#paper-description-format)
- Check papers sections you're interested in (see in [table of contents](#papers-by-topic))
- Maybe search for papers I rated highest: 9/10, 8/10 (usually they have comment on why)
- ~~Do your own summary, star the repo, subscribe~~

## Main Insights

- Data is crucial (highest quality data)
  - EMU from Meta is tuned on just 2k but extremely high quality images
  - Dalle3 report is all about how important is text-image matching in the data
  - Alyosha Efros's talk
  - ... (every 2nd paper/talk)
  - Obvious? Sure, any self-respected ML practitioner learns it in year1, but after hearing it so many time you *feel* it
- Domain experts might be of great help
  - Photography quality labelling (there're agencies who label/relabel smartphone cameras quality by many params, also in EMU domain experts helped to select best images, ...)
  - Mentioned in DeepMind's keynote (in the context of "don't try to blindly apply your 'genious' methods, - consult will it make sense / what is needed / etc")
- Multitask training reduces data requirements by level of magnitude
  - and technically might be equivalent, i.e. does not damage the quality
  - in 2017 or 2019 one of best papers was about ~"which tasks we can combine in multi-task training to improve quality of all"
  - *that might be applicable for huge models though, not for tiny ones
- Self-supervised training: be careful, ensure you don't do something stupid accidentally (e.g. with large batch of text-image pairs using all other pairs as negative examples leads to incorrect negatives)
- DeepMind keynote on project selection
- There's apparently a way of encrypted data inference (so google or whoever is your inference cloud don't know the data you're processing) - not that fast though

## Paper description format

Disclaimer: the notes are biased. Also in many cases I spent very few time on paper so there might be some inaccuracies/mistakes.

- [x/10] (paper main idea description) [Paper title](https://www.youtube.com/watch?v=dQw4w9WgXcQ) my commentary

(some images if idea looked interesting enough and can't be described with a few words and I wasn't lazy)


Ratings are the more the better. Rating is ~usability/novelty of the paper to *me* (read: "very biased"). You can probably Ctrl+F 9/10, 8/10, 7/10, etc

I mostly grouped papers by primary topic, but there're exceptions. e.g. if the only interesting thing in the paper for me was loss I'd put it to losses section regardless of the main topic.

# Workshops

## Video workshop

- Black box interfaces (on ux)
  - chat model is way more convenient for humans.
  - some signals are way easier to provide not with text but image (ref, controlnet, etc)
  - "A good conceptual model let's users predict how input controls affect the output"
  - (just a good question, no great answers I remember) "Low retention rate of GenAI tools, what is missing?"
- Video understanding
  - (tldr) - we really need hierarchical models
  - unsupervised seems to work better than supervised now
  - (historical note) In 2008 was possible to recognize actions like running, sitting down in a car. quite impressive
  - Vid2seq paper can produce dense captions
  - Unsolved video understanding: long term understanding, embodied video understanding (predicting future, potential, likely, interesting, etc)
    - my thoughts:
      - long-term probably needs just some hierarchical model (like different levels of abstraction in summaries).
      - embodied understanding - just learn to predict the future (also should work great in combination with RL/robotics, curiosity, etc)
      - overall does not look that problematic, 1-2 years and we'll be there easy
  - is scaling LLMs the answer? author provide we need 1000x more data/capacity for videos to scale it directly (which is actually just ~20 years in Moor's law). also llms does not capture 4d world complexity (so we need multimodal something).
- AI films (~3-10min movies showcast)
  - are very different
  - in general artists do what before but in a new way, sometimes simpler
  - (have not seen higher [than non-AI] quality works but there should be some)
 
## Continual learning workshop

- Still far from solved, catastrophic forgetting
- minor ideas are to update teacher model with ema student or unfrozen batchnorms - works but not too good

## Quo Vadis / State of Computer Vision

- shortlist of best thoughts
  - Over fitting is because of multiple epochs - let's train on infinite stream of data instead 
  - the word computer in "computer vision" is accidental (vision is central, computers not important in 100 years)
  - New crisis (llms) - focus on creativity instead
- Alyosha Efros's talk (fun to watch, mostly memes, main point - data is king, use good data)
- Lana Lazebnik's talk (on modern science pace, no specific solutions mostly just sharing problems)
  - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/b8370a6e-1e59-47b1-a926-47491cc844f6)
  - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/0273d70e-4bb9-4072-99c3-59923146a972)
  - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/18c56281-56e7-45fd-a895-02e57f72e482)
  - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/b882ebca-ee45-4342-8fa0-faa3d173ba5d)
  - my thoughts (esp after talking with many PhD students on conference) [speculations]:
    - there're 2 kinds of papers - important/fundamental/groundbraking (new problem introduced & solved, completely new level of quality achieved, conceptually new paradigm in solving problems) and incrimental (tiny incrimental improvements in quality, minor hyperparameter change study, datasets exploration)
    - the first type takes a lot of time, in many cases your ideas do not work at all, in some cases (~idea is on the surface) other people publish it faster than you can complete research
    - the second type can be done in really short time, even 1 week start to finish if you try hard. it's sort of not that useful but you'll get your publications/citations/whatever
    - I noticed most PhDs focus on type 1 and fail to publish or focus on type 2 and feel bad about it (or not)
    - probably reasonable strategy is to find balance, spend some time on incremental and some on foundational (split time in week or allocate few months for 1 and few for 2)
    - todo: write type 1 ideas not yet implemented (it was my final todo but I'm too lazy now, maybe will do if repo gets 25+ stars (which it safely won't, right?))
- Antonio Torralba's talk (current LLM crisis -> upcoming CV/entire industry crisis -> what to do with it)
  - great talk full of memes and still valuable
  - basically several lessons from history
    - from most recent: before 2012 people has to know all classic computer vision/ML staff. and still nothing worked with good enough quality for practical problems. now you "stack more layers" and it works. is it bad? people feel old knowledge is useless? not really, many feel excited
    - the greeks theory of extramission (emission) theory as first model of vision
    - the original motivation of images/art is ~"to have wild animals at home w/o it being too dangerous so they step on you during sleep - so someone invented cave painting"
    - at some point in art there was artist who could do perfect realism (or photography). and at this moment some artist thought - what do we do now? important ideas are captured! and when Dali & co comes and draws abstract things and ideas which do not exist in real world. ~go back to original idea of having smth beautiful at home/be able to produce it
    - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/3beaaf4c-f814-45a5-bdfd-2645fd64ca6c)
    - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/4d58ba1a-d871-451c-adb9-7064e18ec494)
    - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/49f9a16f-5e30-4c5b-a146-2554292dcc7b)
    - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/663d5568-3df9-4f6a-8cd6-87f555338ac6)
    - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/c53b9ffc-1ba1-4302-b604-507d16054cef)
    - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/cf69bd11-9e87-4a63-8e43-4aa140b2bcf7)
    - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/1d75e3d7-23b5-4e39-b47f-47125654762d)
    - (comment for last slide^) author provided comparison of number of ~human cognition sensors/cells responsible for vision vs neurons in modeln deep learning - and it's still favorable towards human vision. still mostly a joke as for me but if someone finds natural system easy to build which does not require much training like human vision - that'd be interesting

## Efficient networks

- from big teacher select channels/layers via pruning/etc - still works
- list of current ~sotas ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/fab059e3-49d5-4dba-aa23-d512360d5150)

## Meta GenAI

- emu
  - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/f2350fe1-cb3d-44bc-83ef-c29a4c973300)
  - filter, base model sample and filter again 
  - 16 channel ae important for quality
- text-to-3d
  - good conceptual slide, also you can think on other tasks (x-to-image(done)/3d/video/4d/model/etc)
  - ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/c6c4c30d-3e19-48fd-a7dc-7d7f89ad07cf)

# Keynotes

Robotics training
- LLMs can act like brains of robots (planning agents, etc), but also they can model different users and their different preferences and therefore be a REWARD model as well

Deepmind Research
- I liked the part about problem selection - (how to) choose most impactful thing, generally applicable on other scales as well
- ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/6541073e-44a7-43e7-9ec3-ec36b26e5608)
- ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/470e136e-928f-4817-ac7e-49262023c094)
- ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/34cfd22d-dbf8-487d-ba0b-bab4ec046508)


# Papers by topic

## Diffusion

### Image editing

- [*] [8/10] (to edit real image - generate it & modification with self-attention allowed to view original image)[MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing](https://openaccess.thecvf.com/content/ICCV2023/html/Cao_MasaCtrl_Tuning-Free_Mutual_Self-Attention_Control_for_Consistent_Image_Synthesis_and_ICCV_2023_paper.html) there're few similar attn-mapping methods + extensions to a1111, overall legit idea, works

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/cba98cf3-29e5-4b14-9299-3955ba837f02)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/234cd6d8-4d63-40b8-b9c3-c0b7e63ef6ee)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/8d7eab15-f89c-4ec0-b7b6-626e8a10612f)

- [8/10] (q: how to paste 1 image to another with harmonization (sort of poisson blending problem). a: copy paste noise of inserted image to another noise, map attention masks to the respective locations within the pasted region) [TF-ICON: Diffusion-Based Training-Free Cross-Domain Image Composition](https://openaccess.thecvf.com/content/ICCV2023/html/Lu_TF-ICON_Diffusion-Based_Training-Free_Cross-Domain_Image_Composition_ICCV_2023_paper.html)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/1de96b1e-7e40-45ab-a292-b4e93ce74d4e)


- [6/10] (image manipulation (single image + new prompt -> manipulated). metric to select denoising step for image2image SD automatically - argmax entropy of diffusion training loss on every step. distill edits to other network after that) [Not All Steps are Created Equal: Selective Diffusion Distillation for Image Manipulation](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Not_All_Steps_are_Created_Equal_Selective_Diffusion_Distillation_for_ICCV_2023_paper.html) might be useful metric

### LoRA/Adapters

- [*] [9/10] (database search by (image, edit description), e.g. (img of a train, "at night"). works by textual inversion to S* tokens, but distilled (so any image can get it's token with inference-only)) [Zero-Shot Composed Image Retrieval with Textual Inversion](https://openaccess.thecvf.com/content/ICCV2023/html/Baldrati_Zero-Shot_Composed_Image_Retrieval_with_Textual_Inversion_ICCV_2023_paper.html) models & code [released](https://github.com/miccunifi/SEARLE/releases/tag/weights). *This should have a lot of applications with relatively trivial modifications*. Similar to IP-adapter, just a bit different application scenarious

- [*] [8/10] (what other dimensions you can save in tiny weight part finetuning for big model? precision. so technically for personalized loras you can store them in 1-bit precision as they do in the paper w/o loss in quality) [Revisiting the Parameter Efficiency of Adapters from the Perspective of Precision Redundancy](https://openaccess.thecvf.com/content/ICCV2023/html/Jie_Revisiting_the_Parameter_Efficiency_of_Adapters_from_the_Perspective_of_ICCV_2023_paper.html) what do they even train in 1-bit? +1/-1? for how many weights? technically if the claim is not exploited too much one can save e.g. per user checkpoints with great savings in memory

- [*] [8/10] (diffusion model for faces with relighting) [DiFaReli: Diffusion Face Relighting](https://openaccess.thecvf.com/content/ICCV2023/html/Ponglertnapakorn_DiFaReli_Diffusion_Face_Relighting_ICCV_2023_paper.html) faces are reconstructed really well and indeed only lighting changes. maybe useful for other decompositions

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/2a4362fb-520b-4937-9bdb-3faf3db54a11)

- [6/10] (how to add new modalities encoders to pretrained text2image models? basically you only need paired data of your new modalities and text-image, train small adapter from your modality encoder (can be frozen) for merging with text encoder output before all cross-attentions) [GlueGen: Plug and Play Multi-modal Encoders for X-to-image Generation](https://openaccess.thecvf.com/content/ICCV2023/html/Qin_GlueGen_Plug_and_Play_Multi-modal_Encoders_for_X-to-image_Generation_ICCV_2023_paper.html) good to confirm that simple idea works

- [6/10] (how to tune diffusion with few params - train gamma params (for attn activations and feed forward) - their benchmark showed 8x better quality and slightly more param efficiency than 8/16 rank loras) [DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-efficient Fine-Tuning](https://openaccess.thecvf.com/content/ICCV2023/html/Xie_DiffFit_Unlocking_Transferability_of_Large_Diffusion_Models_via_Simple_Parameter-efficient_ICCV_2023_paper.html) Probably can be used as adapter for sd controlnets as well

- [6/10] (customization. encoder to text embedding from 1 image (+main object mask) + finetuned keys/values for SD attention + extra "local" attention (preserving spatial structure & masking) embedding preserving spatial structure & extra trainable keys/values. during training predicts main + extra tokens, the rest is abandoned on inference as non-object related. ) [ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation](https://openaccess.thecvf.com/content/ICCV2023/html/Wei_ELITE_Encoding_Visual_Concepts_into_Textual_Embeddings_for_Customized_Text-to-Image_ICCV_2023_paper.html) results are not that impressive (e.g. check kitten). something is missing, but spatial attention from image embedding itself makes sense to me ("local mapping") as well as one-shot encoder

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/e4af1db3-6daf-4ecc-940a-1a05ba88cd73)

- [5/10] (set of binary masks, each connected with word in text + prompt -> segmantation-conditioned generation. how: force attention mask for selected words for which exist binary mask to match binary mask via loss, upd z_t iteratively) [Zero-Shot Spatial Layout Conditioning for Text-to-Image Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/html/Couairon_Zero-Shot_Spatial_Layout_Conditioning_for_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html) easier to train controlnet these days, you rarely need only single image edit. but maybe connecting such controlnet with text tokens to force attention may improve quality more

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/6a64af5f-9435-4d30-9d1e-b00e3171d863)

### Enforce prompt matching

- [9/10] (q: how to fix problem that some part of prompt is ignored? e.g. for frog in crown you get just frog. a: you need to fix attention (on finer steps it's disappearing despite initially being present; to do that they introduced losses which adjust z_t during generation)[A-STAR: Test-time Attention Segregation and Retention for Text-to-image Synthesis](https://openaccess.thecvf.com/content/ICCV2023/html/Agarwal_A-STAR_Test-time_Attention_Segregation_and_Retention_for_Text-to-image_Synthesis_ICCV_2023_paper.html)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/847720fc-18c9-4dc1-96d8-5dfa54b6fabc)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/e7334421-2e5f-441e-82dd-45f8b19eecee)

- [8/10] (~textual inversion for exclusive sets of attributes, e.g. gender, skintone, etc by image references. but not with actual textual inversion but by clip embedding optimization similar to stylegan-nada) [ITI-GEN: Inclusive Text-to-Image Generation](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_ITI-GEN_Inclusive_Text-to-Image_Generation_ICCV_2023_paper.html) you can generate "man with glasses" but you can't generate "man without glasses" (usually negative prompts don't guarantee that, esp if you generate thousands of images) so that work is useful for controllable generation

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/606408ea-932f-417b-b2f4-1e2f6aaf8ac8)

### Other / better guidance

- [*] [8/10] (better claffifier (not free) guidance - backprop to all noises consequitively from original image. super slow. but quality is better) [End-to-End Diffusion Latent Optimization Improves Classifier Guidance](https://openaccess.thecvf.com/content/ICCV2023/html/Wallace_End-to-End_Diffusion_Latent_Optimization_Improves_Classifier_Guidance_ICCV_2023_paper.html) should also work to any losses (segmentation, identity, etc) since explicit gradient is used. isn't this obvious idea though? too obvious even, I'm surprised clf-guidance was done w/o full denoising, only issue was gradient backward through huge network on all steps so they reformulate it as invertible diffusion here

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/3712835f-441c-45f8-8e25-306a1103f84d)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/5bbbc738-606a-471e-8e3d-19b600211ed0)

- [5/10] (cfg-like guidance in order to improve sampling quality (~details) - basically reinforces effect of attention vs no use of attention ~attn>threshold mask) [Improving Sample Quality of Diffusion Models Using Self-Attention Guidance](https://openaccess.thecvf.com/content/ICCV2023/html/Hong_Improving_Sample_Quality_of_Diffusion_Models_Using_Self-Attention_Guidance_ICCV_2023_paper.html) some sampling quality boost for marginal cost increase. [code](https://github.com/KU-CVLAB/Self-Attention-Guidance)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/d7e6b43c-b51d-4f54-a2b6-f7d29183f29b)

### Domain adaptation

- [*] [9/10] (domain adaptation (for style) on few images w/o finetuning: sample from style-specific noise distribution! simple per-pixel mean/stds used in their method. results look impressive and work on just a few images) [Diffusion in Style](https://openaccess.thecvf.com/content/ICCV2023/html/Everaert_Diffusion_in_Style_ICCV_2023_paper.html)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/b5c5973e-40e0-43b9-83f9-2cbcb94d3d2c)

- [6/10] (turns out 2 *stochastic* diffusion models, trained independantly, given same "seed" produce related images (!!! lol) -> in this work they are generating surprasingly good paired images from 2 models / edits by prompt modification from single model. results look good)[A Latent Space of Stochastic Diffusion Models for Zero-Shot Image Editing and Guidance](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_A_Latent_Space_of_Stochastic_Diffusion_Models_for_Zero-Shot_Image_ICCV_2023_paper.html) mostly interesting theory since there's no community interest -> wide adoption in these models for now 

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/3330c11a-0938-4bb9-94fa-349540813d04)

### Removing/modifying concepts in pretrained diffusion

- [8/10] (problem: want to change diffusion assumption on the prompt (e.g. messi -> playing basketball not football, roses -> are blue). solution: given original/edited prompt modify text cross-attn layers to give similar masks for prompt1 to prompt2 -> update these params of the model -> updated model always thinks new behaviour is correct since layers are fused) [Editing Implicit Assumptions in Text-to-Image Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/html/Orgad_Editing_Implicit_Assumptions_in_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html) aka TIME. that's probably better way to patch the model instead of just stripping it away from all knowledge like anti-dreambooth, etc

- [6/10] (remove concept C by forcing model to produce same noise as ok concept C', e.g. "grumpy cat"->"cat". side-effect - can preserve individual concepts while removing combinations ("kids with guns"->"kids", but "kids" and "guns" separetely still works)[Ablating Concepts in Text-to-Image Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/html/Kumari_Ablating_Concepts_in_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html) probably most practical and easy to use. although the one below ("Erasing Concepts from Diffusion Models") in theory preserves the knowledge of the concept, just does not generate it by prompt directly (which can be good as it keeps more knowledge and bad as... it keeps this knowledge which still can be tampered with other prompts)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/4096a632-be66-4339-902b-ae42f563b143)

- [5/10] (see poster explanations. basically use frozen model & tuned one, in tuned one use cfg-like guidance to guide in opposite direction from frozen for selected concepts) [Erasing Concepts from Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/html/Gandikota_Erasing_Concepts_from_Diffusion_Models_ICCV_2023_paper.html) looks like better idea compared to anti-dreambooth

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/0fc1c1b0-300b-4e10-b10b-596d789d3af0)

- [3/10] (protected images/styles -> tune dreambooth to predict noise on them) [Anti-DreamBooth: Protecting Users from Personalized Text-to-image Synthesis](https://openaccess.thecvf.com/content/ICCV2023/html/Van_Le_Anti-DreamBooth_Protecting_Users_from_Personalized_Text-to-image_Synthesis_ICCV_2023_paper.html) erasing should not work like this - it's damaging the model. at least let the model produce some plausable images

### Not just text2image

- [*] [9/10] (joint image+segmentation map generation by reformulated noise distribution) [Learning to Generate Semantic Layouts for Higher Text-Image Correspondence in Text-to-Image Synthesis](https://openaccess.thecvf.com/content/ICCV2023/html/Park_Learning_to_Generate_Semantic_Layouts_for_Higher_Text-Image_Correspondence_in_ICCV_2023_paper.html) this is INSIGHTFUL paper. basically they do some math to show that joint generation is equivalent to separate. `insight is that you need MUCH LESS data because you predict multiple things together`. e.g. for generative models on videos, 3d, etc difficult problems (more difficult than just images) should be very helpful

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/205c3159-6827-4931-922d-f856dbbf3131)

- [5/10] (paired dataset included) [Generating Realistic Images from In-the-wild Sounds](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Generating_Realistic_Images_from_In-the-wild_Sounds_ICCV_2023_paper.html)

- [4/10] (text/image-to-text/image) [Versatile Diffusion: Text, Images and Variations All in One Diffusion Model](https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Versatile_Diffusion_Text_Images_and_Variations_All_in_One_Diffusion_ICCV_2023_paper.html)

- [4/10] (train unconditional diffusion + conditional (on 1 view, targets obtained throigh warp). 360 views by iterative inference) [3D-aware Image Generation using 2D Diffusion Models](https://jeffreyxiang.github.io/ivid/) maybe useful proxy for something, but not very practical. quality of warped targets likely low

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/4c80e5ea-f8c1-45b4-aa3a-1d321dd06eb5)

### Security risks

- [7/10] (add backdoor to TEXT ENCODER to poison ANY text2image model trained on that. cyrillic "o" is invisible even for humans attack) [Rickrolling the Artist: Injecting Backdoors into Text Encoders for Text-to-Image Synthesis](https://openaccess.thecvf.com/content/ICCV2023/html/Struppek_Rickrolling_the_Artist_Injecting_Backdoors_into_Text_Encoders_for_Text-to-Image_ICCV_2023_paper.html) most of current text2image models are based on clip, so if somehow official checkpoint will be hacked all the models will also get hacked. bad prompt filtering pipelines should probably check for such attacks now before inference though. now what is really interesting - maybe instead of injecting backdoors they're already there - what if someone can find "abirvalg" - some weird combination of tokens which activates backdoor mode. sort of like "Try to impersonate DAN" attack for LLMs but prompt has to be optimized. what the finding of the paper tells is that the found magic word would affect all models trained with such encoder

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/4b30209f-10c5-475d-8393-9a03965075d5)

- [6/10] (imagine you release model with invisible watermarking. if someone infers directly that model - you probably detect it reliably. if someone finetunes it - your watermarking is mostly useless. in this work they added extra loss for close params to also produce watermarked models, sort of GAN game) [Towards Robust Model Watermark via Reducing Parametric Vulnerability](https://openaccess.thecvf.com/content/ICCV2023/html/Gan_Towards_Robust_Model_Watermark_via_Reducing_Parametric_Vulnerability_ICCV_2023_paper.html)

### Faster inference

- [*] [7/10] (problem: pretrained t2i model -> how to infer fast and not loose quality. solution: look for non-uniform steps + subnetwork -> define search space -> evolutionary search (w/o retraining, only inference) with FID eval -> good results) [AutoDiffusion: Training-Free Optimization of Time Steps and Architectures for Automated Diffusion Model Acceleration](https://openaccess.thecvf.com/content/ICCV2023/html/Li_AutoDiffusion_Training-Free_Optimization_of_Time_Steps_and_Architectures_for_Automated_ICCV_2023_paper.html) results too good for such simple method... didn't check details but interesting how they not loose much quality by abandoning some parts of architecture

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/9bd02163-cd76-4c71-82df-f28efe49e083)

### Unsorted

- [7/10] (dataset for gt attribution via diffusion customization -> eval existing approaches -> tune clip/etc -> clip(generated)@clip(ref) -> estimate attribution. quality of attibution predictor trained is surprisingly good [at least high matches are super relevant images]) [Evaluating Data Attribution for Text-to-Image Models](https://peterwang512.github.io/GenDataAttribution/) only issue is that it just compares images, it does not know if it was in training or not. but e.g. artist compensation is possible based on that

- [6/10] (sort of bayes decomposition and basis is discovered automatically. learned concepts can be combined) [Unsupervised Compositional Concepts Discovery with Text-to-Image Generative Models](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Unsupervised_Compositional_Concepts_Discovery_with_Text-to-Image_Generative_Models_ICCV_2023_paper.html) the idea itself is cool, but if I understand coerrectly number of concepts is hyperparameter

- [4/10] (argmin|eps_pred(img of class_i)-eps|) [Your Diffusion Model is Secretly a Zero-Shot Classifier](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.html) smart, but not very useful

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/0f7efc56-364c-4a61-a4f5-7f90af6ff886)

## Data

### Investigations

- [5/10] (findings - 1)prompts on average lead to India/US as most relevant 2)adding country into the prompt improves generation but not completely sufficient 3)dalle2 likely had much better filtration than SD because for (2) they have bigger improvement)[Inspecting the Geographical Representativeness of Images from Text-to-Image Models](https://openaccess.thecvf.com/content/ICCV2023/html/Basu_Inspecting_the_Geographical_Representativeness_of_Images_from_Text-to-Image_Models_ICCV_2023_paper.html) that's important research but sometimes it's surprising how you can publish on top venues just investigating the data

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/c89943de-38ee-444c-a977-88104107f0cf)

- [3/10] (some biases of models, ~light skin -> more feminine, etc) [Beyond Skin Tone: A Multidimensional Measure of Apparent Skin Color](https://ai.sony/publications/Beyond-Skin-Tone-A-Multidimensiona-Measure-of-Apparent-Skin-Color/)

### Dataset compression to N samples

- [2/10] (models trained on data-distilled to few samples are overconfident -> need "calibration" (more reasonable logit distribution) -> some fixes suggested in this paper) [Rethinking Data Distillation: Do Not Overlook Calibration](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Rethinking_Data_Distillation_Do_Not_Overlook_Calibration_ICCV_2023_paper.html) since original problem (compressing dataset to 100 samples) is still not useful (quality is bad, generalization beyound cifar100 is unlikely), adjustments are also not helpful

### Datasets

- [*] [6/10] (dataset of 10k artifacts segmentation from various generative models - GANs/Diffusion. also trained segmentation & inpainting but not code yet) [Perceptual Artifacts Localization for Image Synthesis Tasks](https://owenzlz.github.io/PAL4VST/) probably useful but not sure about quality, esp on new types of images

- [5/10] (dataset with 5k diverse photos from smartphones estimated by experts on quality metrics) [An Image Quality Assessment Dataset for Portraits](https://openaccess.thecvf.com/content/CVPR2023/papers/Chahine_An_Image_Quality_Assessment_Dataset_for_Portraits_CVPR_2023_paper.pdf) that's cvpr paper but company had a booth and advertised it. research-only license, terms probably tricky, idea to get such labelling from experts is worthy though. I talked with them a little bit - important thing for quality estimations is to recompute benchmarks (cameras becoming better and better so there's no "perfect" quality in gt) every 1-2 years at least. camera producers usually go to these agencies to measure their quality (ratings are open although I'm not sure customers actually visit such websites to check)

### Synthetic labels

- [4/10] (self-explanatory title. quality claimed to be ok on real data and sota on zero-shot approaches) [DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_DiffuMask_Synthesizing_Images_with_Pixel-level_Annotations_for_Semantic_Segmentation_Using_ICCV_2023_paper.html) labelling is ofc superior if you have resources. probably same approach can be useful in some other problems

- [4/10] (optimization-based refinement of mask from attention, tune diffusion & update mask. expensive, but claim sota zero-shot) [Foreground-Background Separation through Concept Distillation from Generative Image Foundation Models](https://openaccess.thecvf.com/content/ICCV2023/html/Dombrowski_Foreground-Background_Separation_through_Concept_Distillation_from_Generative_Image_Foundation_Models_ICCV_2023_paper.html)

## Multi-modality

### VLMs (VQA, captioning, zero-shot classification, etc)

- [8/10] (~synth data from llms, learned classifier on clip text embedding when infer on image clip embedding. close to sota on captioning, vqa, destroyed sota on some unpopular tasks) [I Can't Believe There's No Images! Learning Visual Tasks Using only Language Supervision](https://openaccess.thecvf.com/content/ICCV2023/html/Gu_I_Cant_Believe_Theres_No_Images_Learning_Visual_Tasks_Using_ICCV_2023_paper.html) I think I've seen other similar work where there's extra adapter image embedding -> text embedding and it should work even better.

- [5/10] (attempt to cure ood hallusinations on captioning for VLMs) [Transferable Decoding with Visual Entities for Zero-Shot Image Captioning](https://openaccess.thecvf.com/content/ICCV2023/html/Fei_Transferable_Decoding_with_Visual_Entities_for_Zero-Shot_Image_Captioning_ICCV_2023_paper.html) sophisticated bert-like method to avoid overfit and allow ood generalization. maybe it's a better idea to just have ood examples to bring it in-domain?

### CLIP Training

- [8/10] (equivariant here ~= text-image scores are proportial to actual relevance, i.e. 0.7 is meaningful. given 2 image-text pairs [semantically similar! ~= not too different] they design simple losses e.g. similarity(text_1, image_2)==similarity(text_2, image_1), see others below) [Equivariant Similarity for Vision-Language Foundation Models](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Equivariant_Similarity_for_Vision-Language_Foundation_Models_ICCV_2023_paper.html) labelling is even more crucial for such alignment training. can be combined with other clip optimizations in training (e.g. filtering "hard samples" which are technically valid pairs)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/5c648eed-2407-437a-ad32-b71f2ace7c49)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/de1783ef-12db-4b6d-b284-2e6d2a807a0c)

- [7/10] (affinity mimicking = same distribution of text-image similarity on train batch, weight inheritance = choose teacher weights part) [TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_TinyCLIP_CLIP_Distillation_via_Affinity_Mimicking_and_Weight_Inheritance_ICCV_2023_paper.html) smaller/faster clip models are useful by themselves when performance matters. side note: progressive distillation works better (e.g. 100%->25% capacity is worse than 100->50->25 for same training time)

### Prompt tuning

- [5/10] (prompt learning for classification with couple extra losses) [Self-regulating Prompts: Foundational Model Adaptation without Forgetting](https://openaccess.thecvf.com/content/ICCV2023/html/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.html)

### CLIP zero-shot quality improvements

- [6/10] (llm generates prompts per every class need to be detected. q for llm: "what does [class_name] look like?". claims to be better than hand-designed (well it scales up easily true)) [What Does a Platypus Look Like? Generating Customized Prompts for Zero-Shot Image Classification](https://openaccess.thecvf.com/content/ICCV2023/html/Pratt_What_Does_a_Platypus_Look_Like_Generating_Customized_Prompts_for_ICCV_2023_paper.html) there were some other work saying that writing "[random characters] [class_name]" gives better accuracy than LLM-designed ones (averaged among these random characters ofc)

- [6/10] (learn N "style" text embeddings ~ "a S_i style of [object]" where object is dog/cat/etc classes. styles are not supervised on anything, only text encoder and no images are used. on top of learned style augmented prompts linear classifier is trained. in the end argmax clip mean "a s_i style of [class_name]". works very good) [PromptStyler: Prompt-driven Style Generation for Source-free Domain Generalization](https://openaccess.thecvf.com/content/ICCV2023/html/Cho_PromptStyler_Prompt-driven_Style_Generation_for_Source-free_Domain_Generalization_ICCV_2023_paper.html) again I remember work with image2text clip embedding adapter with similar idea on this conference - should work even better. does not lead to interesting text2image styles, but probably some modifications can help finding interesting ones automatically (although with visual feedback should be better)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/4dbbae7f-04ef-4abe-bbc1-1c9153cac9e3)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/49f4b853-e959-4fc6-8fb0-2fb36ffde1fd)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/f45df084-e0b0-4464-aefc-561f782977fa)

### CLIP Inference

- [6/10] (draw red circle around smth -> see what clip predicts where among the proposed variants. man->criminal, woman->missing. works for landmarks/etc) [What does CLIP know about a red circle? Visual prompt engineering for VLMs](https://openaccess.thecvf.com/content/ICCV2023/html/Shtedritski_What_does_CLIP_know_about_a_red_circle_Visual_prompt_ICCV_2023_paper.html) that's nice exploit. direct practical usage is unclear, though on the meta-level should be applicable to other models

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/6bc69a61-464f-4261-890e-2068fdae7e8f)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/91a08e69-b367-42f4-a9ec-83f1b65d58e3)

### CLIP Data/abilities

- [9/10] (that's some unknown paper from conference) (0 shot better - for contrastive pretraining because of ambiguity and large batch might be good img-text negative pairs (i.e. not on batch diagonal), so instead they consider 3 similarities between img/text/ij - imgs, texts, img-texti for the right loss) very simple/obvious idea yet very helpful ![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/8ddec19a-e222-4980-929a-fef90c836e09)

- [5/10] (just finetune clip on data with number or objects in captions) [Teaching CLIP to Count to Ten](https://openaccess.thecvf.com/content/ICCV2023/html/Paiss_Teaching_CLIP_to_Count_to_Ten_ICCV_2023_paper.html)

- [3/10] (basically tuned clip on negative image-text pairs by adding "no" to prompt, e.g. "image of NO cat" with dog image) [CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_CLIPN_for_Zero-Shot_OOD_Detection_Teaching_CLIP_to_Say_No_ICCV_2023_paper.html)

## GANs

### Latent space manipulations

- [7/10] (connect w to selected regions (hardset or segmentation) by adding loss for this during training) [LinkGAN: Linking GAN Latents to Pixels for Controllable Image Synthesis](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_LinkGAN_Linking_GAN_Latents_to_Pixels_for_Controllable_Image_Synthesis_ICCV_2023_paper.html) likely not very practical, interesting attributes are heavily entangled (if you change one, another have to change). nice idea though

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/102ede02-f4c0-43b8-acfc-975f6cad2fd1)

- [4/10] (gan features + face parameters (expression, lighting, pose) -> learn mapping to "better" space. cherry picked examples in paper are better than styleflow) [Conceptual and Hierarchical Latent Space Decomposition for Face Editing](https://openaccess.thecvf.com/content/ICCV2023/html/Ozkan_Conceptual_and_Hierarchical_Latent_Space_Decomposition_for_Face_Editing_ICCV_2023_paper.html) didn't check details

- [3/10] (find latent directions in gans, ~stylegans) [Householder Projector for Unsupervised Latent Semantics Discovery](https://openaccess.thecvf.com/content/ICCV2023/html/Song_Householder_Projector_for_Unsupervised_Latent_Semantics_Discovery_ICCV_2023_paper.html) found directions look entangled as usual. didn't check details

### Domain adaptation

- [*] [7/10] (stylegan-nada improved. idea: instead of single text direction use multiple ones and match distributions of image & text directions. 1)find multiple directions ~close to original trg text embedding + most dissimilar from one another -> ~uniformly distributed some distance around the embedding of trg text 2)for image-image directions in training batch and text-text directions (original and all augmented) penalize both mean and covariance mismatch. see formulas below for details) [Improving Diversity in Zero-Shot GAN Adaptation with Semantic Variations](https://openaccess.thecvf.com/content/ICCV2023/html/Jeon_Improving_Diversity_in_Zero-Shot_GAN_Adaptation_with_Semantic_Variations_ICCV_2023_paper.html) stylegan-nada is sort of simple and naive loss, there're many small changes you could propose to improve results (see e.g. [this work](https://arxiv.org/abs/2110.08398) for clip within loss)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/a72220c7-1795-419f-a95f-1721525e3e3e)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/21598e68-b8c0-473c-9285-baa8ac137157)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/ac40e84c-aa0d-4328-99fb-a87d4e558834)

- [6/10] (photo->avatar. where avatar is parametrized model, i.e. hair/ear/eyebrow/etc params. train 2 unconditional generator (real faces & parameters of avatars) after which mapping between them. they had small paired dataset of 400 imgs, hand-crafted by artist on volunteer selfies) [Cross-modal Latent Space Alignment for Image to Avatar Translation](https://openaccess.thecvf.com/content/ICCV2023/html/de_Guevara_Cross-modal_Latent_Space_Alignment_for_Image_to_Avatar_Translation_ICCV_2023_paper.html) random thought: maybe it's possible to learn alignment between 2 different generators (like 2 face GANs -> learn mapping from first GAN latent space to second GAN latent space. only need some paired data)

- [5/10] (gan finetuning to dissimilar domain. 2 ideas: 1)~regularize init vs tuned feature discribution ("smoothness") 2)multi-resolutioal patch D) [Smoothness Similarity Regularization for Few-Shot GAN Adaptation](https://openaccess.thecvf.com/content/ICCV2023/html/Sushko_Smoothness_Similarity_Regularization_for_Few-Shot_GAN_Adaptation_ICCV_2023_paper.html) results on 10 imgs are still crap but *less crap* (in more reasonable problems should help as well). haven't found comparison vs augmentations (stylegan2-ada, etc) - usually that helped a lot)

### Unsorted

- [6/10] (reason: statistical assumption of stylegan2 is too strong [the one that we do not need subtract mean in weight demodulation], which leads to some ood features [high values] ) [Feature Proliferation -- the "Cancer" in StyleGAN and its Treatments. they can both detect [heuristic] and fix [rescaling affected areas] the issue. both are cheap, see formulas](https://openaccess.thecvf.com/content/ICCV2023/html/Song_Feature_Proliferation_--_the_Cancer_in_StyleGAN_and_its_Treatments_ICCV_2023_paper.html)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/d46db17a-151e-4a2a-8097-357c739de058)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/dc7393c5-342e-4bd5-8f6a-b34278653324)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/24598572-aaed-4c62-94bd-987fa781c8dd)

## Training improvements

### Training

- [6/10] (train bigger model from smaller) [TripLe: Revisiting Pretrained Model Reuse and Progressive Learning for Efficient Vision Transformer Scaling and Searching](http://openaccess.thecvf.com//content/ICCV2023/papers/Fu_TripLe_Revisiting_Pretrained_Model_Reuse_and_Progressive_Learning_for_Efficient_ICCV_2023_paper.pdf) sort of curriculum learning but for weights. might be a bit useful if you go to ridiculously large models (GPT5 or something)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/cb0d71b6-b95d-4574-a230-89a5aa6dfdee)

- [5/10] (lowres -> highres & increase strength of augs. downsample is done with cropping in freq domain [claim is that's more precise/justified]. training cost of huge model trained from scratch reduced ~20-30%) [EfficientTrain: Exploring Generalized Curriculum Learning for Training Visual Backbones](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_EfficientTrain_Exploring_Generalized_Curriculum_Learning_for_Training_Visual_Backbones_ICCV_2023_paper.html) it'd be more curious how it generalizes to LLM/other foundational models. likely useless for finetuning. training on lowres first is beyound obvious as well as increasing strength of augmentations, still maybe take practical tips from here. for augmentations they use [RandAug](https://paperswithcode.com/paper/randaugment-practical-data-augmentation-with) with progressive strengths. frequency domain also can be explored more during training (e.g. more losses, etc).

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/777d5845-9cf6-420b-bb45-7ce7c98157d5)

- [5/10] (7x faster imagenet training ~any model. stored 400 augs per image with 10model-ensemble average predictions. training code 2 lines change) [Reinforce Data, Multiply Impact: Improved Model Accuracy and Robustness with Dataset Reinforcement](https://arxiv.org/abs/2303.08983) thank you. not really often common people need to train smth on imagenet from scratch but might be useful eventually

### Finetuning/other task adaptation

- [3/10] (continual learning - pretrain base transformer -> adapt per task with conv transforming attn weights) [Exemplar-Free Continual Transformer with Convolutions](https://openaccess.thecvf.com/content/ICCV2023/html/Roy_Exemplar-Free_Continual_Transformer_with_Convolutions_ICCV_2023_paper.html) this is also for class-incremental setup (task id is not known - which is ?stupid because in practice it's known always. well at least [chatgpt didn't provide reasonable answer](https://chat.openai.com/share/bb95eb99-f6cd-416c-be0c-e37c7da003b4))

### Decoding

- [3/10] (autoregressive decoder with k tokens/inference step for image compression. show that predefined token sampling schedule perform as well or better than random (how's that not obvious though?)) [M2T: Masking Transformers Twice for Faster Decoding](https://openaccess.thecvf.com/content/ICCV2023/html/Mentzer_M2T_Masking_Transformers_Twice_for_Faster_Decoding_ICCV_2023_paper.html) no new insights

### Losses

- [7/10] (simple tasks but on very high resolution - try to make realtime. idea1: overparametrization (10 branches later result to single 5x5 conv on inference), idea2: lightweighted feature fusion (f1*f2+bias), idea3: outlier-aware loss (prevents blur of l2 loss)) [SYENet: A Simple Yet Effective Network for Multiple Low-Level Vision Tasks with Real-Time Performance on Mobile Device](https://openaccess.thecvf.com/content/ICCV2023/html/Gou_SYENet_A_Simple_Yet_Effective_Network_for_Multiple_Low-Level_Vision_ICCV_2023_paper.html) results are not super impressive but maybe some of that can be useful (OOL/overparametrization)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/ada1e608-dad1-497d-84df-ca51f91c3b74)

- [6/10] (1-img stylization preserving content. for content preservation patch contrastive loss from eps-predictor of diffusion [so full denoising / noise-aware models not required]) [Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Zero-Shot_Contrastive_Loss_for_Text-Guided_Diffusion_Image_Style_Transfer_ICCV_2023_paper.html) quality is complete crap (which is strange, probably they should not do it in patch-contrastive way because features from different regions of images are often similar?) but general idea of using eps-predictor as sort-of perceptual similarity source/target looks legit, should be applicable for other problems (definetely applied somewhere though?)

- [6/10] (forced classifier to attend the right place of the image and it improved quality and *maybe* reliability) [Studying How to Efficiently and Effectively Guide Models with Explanations](https://openaccess.thecvf.com/content/ICCV2023/html/Rao_Studying_How_to_Efficiently_and_Effectively_Guide_Models_with_Explanations_ICCV_2023_paper.html)

### Federated learning

- [*] [8/10] (how to adapt classifier per every user to improve quality. user side setup: clf(model(img, prompt)) where prompt is just a few trainable params, model is frozen (e.g. pretrained foundational model), clf is local per-client classifier. server setup: base prompts, prompt generator network (user descriptor -> better user prompt). rough idea: when training starts baseline zero-shot performance already works somehow, every training step you don't have GT but use current inference prediction as GT to update the system. so every step on user side clf, prompt are updated and on backend side prompt generator is updated) [Efficient Model Personalization in Federated Learning via Client-Specific Prompt Generation](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Efficient_Model_Personalization_in_Federated_Learning_via_Client-Specific_Prompt_Generation_ICCV_2023_paper.html) some variation for personalized text2image models?

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/8401a8da-2379-470d-a1ec-a30d006c0a8a)

## Architectures

### Tricks

- [6/10] (heuristic to process only part of tokens in ViT - the important/difficult ones. in practice can be ~25% but depends on task) [A-ViT: Adaptive Tokens for Efficient Vision Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Yin_A-ViT_Adaptive_Tokens_for_Efficient_Vision_Transformer_CVPR_2022_paper.pdf)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/e3cec60b-7d9b-4333-a38e-5443cf63ef1a)

- [5/10] (same trick with skipping most tokens, for videos) [Eventful Transformers: Leveraging Temporal Redundancy in Vision Transformers](https://openaccess.thecvf.com/content/ICCV2023/papers/Dutson_Eventful_Transformers_Leveraging_Temporal_Redundancy_in_Vision_Transformers_ICCV_2023_paper.pdf)

- [5/10] (~~abandon some tokens~~ -> focus attention on some tokens) [Less is More: Focus Attention for Efficient DETR](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Less_is_More_Focus_Attention_for_Efficient_DETR_ICCV_2023_paper.html) claims to have better quality/speed tradeoff than abandoning

- [3/10] (normally early exit idea on easy samples does not work / performance degrade, their solution marginally improve it) [Dynamic Perceiver for Efficient Visual Recognition](https://openaccess.thecvf.com/content/ICCV2023/html/Han_Dynamic_Perceiver_for_Efficient_Visual_Recognition_ICCV_2023_paper.html) mostly added to have ref for early exit approach and that it does not really work well

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/0a369b96-0eaa-4e33-8f37-adad04fbd138)

### Attention

- [9/10] (instead of memory-augmented attention [=learnable extra keys and values] they reuse keys and values from the previous N training samples. motivation is that this should better focus on individual samples instead of being beneficial for the entire dataset on average. note that this is *actual memory* - previous outputs/thoughts of the network. to not store too many memories they use k-means centers. memories are updated every N training batches with k-means again) [With a Little Help from your own Past: Prototypical Memory Networks for Image Captioning](https://arxiv.org/abs/2308.12383) memory should be very useful for other generative tasks, maybe not the approach itself but idea at least. like hashgrid encodings in nerfs, some sort of memory for the network to be able to operate, not just extract everything from input & biases, that's very reasonable

- [8/10] (better linear attention. linear attn has quality drop, so they investigate issues and fix them. Y=phi(Q)phi(K)V + depthwise(V) where phi(x)=||x||*(x^p)/||x^p|| where x^p is elementwise power p) [FLatten Transformer: Vision Transformer using Focused Linear Attention](https://openaccess.thecvf.com/content/ICCV2023/html/Han_FLatten_Transformer_Vision_Transformer_using_Focused_Linear_Attention_ICCV_2023_paper.html)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/75f17e85-575d-4dda-b655-5c12db6ec518)

- [6/10] (relu linear attention fixes with depthwise convs) [EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.html)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/216246b6-1ad5-47de-a061-450641618887)

- [6/10] (attend with different level of detail, based on network own attn map. claims sota of the time with some margin) [SG-Former: Self-guided Transformer with Evolving Token Reallocation](https://openaccess.thecvf.com/content/ICCV2023/html/Ren_SG-Former_Self-guided_Transformer_with_Evolving_Token_Reallocation_ICCV_2023_paper.html) the idea makes sense, not sure how efficient the implementation is though

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/6d3a240a-8586-4d73-a27a-461b48948ced)

- [2/10] [Gramian Attention Heads are Strong yet Efficient Vision Learners](https://openaccess.thecvf.com/content/ICCV2023/html/Ryu_Gramian_Attention_Heads_are_Strong_yet_Efficient_Vision_Learners_ICCV_2023_paper.html) no benefit for now, theory + work on par

### Modules/Layers

- [7/10] (1d oriented convs with efficient cuda implementation -> quality ~same as 2d on some tasks -> receptive field is bigger) [Convolutional Networks with Oriented 1D Kernels](https://openaccess.thecvf.com/content/ICCV2023/html/Kirchmeyer_Convolutional_Networks_with_Oriented_1D_Kernels_ICCV_2023_paper.html) on-device efficiency is always questionable for new layers, but idea is interesting. maybe some combination of 2d & 1d-oriented is better. e.g. 1d-oriented with huge kernel sizes to allow good receptive field, but 1 time per block only. or might be interesting to predict orientation first (per-pixel) and when use it similar to attention (although not sure it will be efficient this way)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/e013ad62-83c0-4031-877f-e50d085095fe)

- [6/10] (compression) [Shortcut-V2V: Compression Framework for Video-to-Video Translation Based on Temporal Redundancy Reduction](https://openaccess.thecvf.com/content/ICCV2023/html/Chung_Shortcut-V2V_Compression_Framework_for_Video-to-Video_Translation_Based_on_Temporal_Redundancy_ICCV_2023_paper.html) maybe one interesting idea - they use deformable convs architecture, might be suitable for other video tasks

- [6/10] (similar to SE block with better accuracy claimed, no params for visual transformers: y = x/2+x.mean(channel_dim)/2. motivation: investigation vits found that they try to learn dense (~=close to uniform per tokens) attention, despite it's hard to learn due to high gradients in these areas. the proposed module is explicit parameter-free extreme dense attention (=uniform attention)) [Scratching Visual Transformer's Back with Uniform Attention](https://openaccess.thecvf.com/content/ICCV2023/html/Hyeon-Woo_Scratching_Visual_Transformers_Back_with_Uniform_Attention_ICCV_2023_paper.html)

- [5/10] (~inception architecture but with attention) [Scale-Aware Modulation Meet Transformer](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_Scale-Aware_Modulation_Meet_Transformer_ICCV_2023_paper.html) marginal improvements, likely speed downgrade

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/0a2dcbb2-fa1b-4b1d-83dd-14756c425a53)

- [5/10] (inside inverted bottleneck some efficient attn operator) [Rethinking Mobile Block for Efficient Attention-based Models](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Rethinking_Mobile_Block_for_Efficient_Attention-based_Models_ICCV_2023_paper.html) probably useful didn't check details

- [4/10?] (multiple feature aggregation modules) [Building Vision Transformers with Hierarchy Aware Feature Aggregation](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Building_Vision_Transformers_with_Hierarchy_Aware_Feature_Aggregation_ICCV_2023_paper.html) only gave moment look but marginal improvements, not fast implementation? (e.g. clustering is parameter-free but not instant)

### Downsample/upsample

- [5/10] (generalization of all pooling algorithms, trainable, better quality, ~slower, targeted for final pooling before classification, gives better attention maps) [Keep It SimPool: Who Said Supervised Transformers Suffer from Attention Deficit?](https://openaccess.thecvf.com/content/ICCV2023/html/Psomas_Keep_It_SimPool_Who_Said_Supervised_Transformers_Suffer_from_Attention_ICCV_2023_paper.html)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/d7909990-4209-45f6-a59e-39e37016fd5e)

- [5/10] (tiny resizer (downsample). claimed to be useful for classification/segmentation to work on high resolution) [MULLER: Multilayer Laplacian Resizer for Vision](https://openaccess.thecvf.com/content/ICCV2023/html/Tu_MULLER_Multilayer_Laplacian_Resizer_for_Vision_ICCV_2023_paper.html) on device passing large textures to gpu is expensive operation so not sure how useful it is in practice

- [5/10] (light upsample for dense predictions (but not superres)) [Learning to Upsample by Learning to Sample](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.html) likely marginal improvements in real life if any

- [4/10] (spectral space pooling, claims sota quality on classification/segmentation) [SPANet: Frequency-balancing Token Mixer using Spectral Pooling Aggregation Modulation](https://openaccess.thecvf.com/content/ICCV2023/html/Yun_SPANet_Frequency-balancing_Token_Mixer_using_Spectral_Pooling_Aggregation_Modulation_ICCV_2023_paper.html) not sure it's actually fast

- [4/10?] (flexible downsample with non-integer stride) [FDViT: Improve the Hierarchical Architecture of Vision Transformer](https://openaccess.thecvf.com/content/ICCV2023/html/Xu_FDViT_Improve_the_Hierarchical_Architecture_of_Vision_Transformer_ICCV_2023_paper.html) only gave moment look but 0.4 improvement is questionable, flexible downsample is unlikely fast as well

### Misc architectures

- [5/10] (DiT; unet->transformer leads to much more scaleble architecture (both up and down) & claims better FID with same flops)[Scalable Diffusion Models with Transformers](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html) why meta/stability didn't use transformers in EMU/sdxl then? or any other work using transformer instead of unet? (tldr from Slavchat discussion - there's [some proof](https://twitter.com/ylecun/status/1717676624865382901?s=46&t=cPwUpELiKQDigRVu8sbERw) that what is important is computation - it should give +- same quality regardless of architecture (with reasonable architectures). in theory the benefit of transformers is that they're more easily scaleble. Kudos to Michael, Vadim, Seva, Aleksandr, George)

- [4/10] [Masked Autoencoders Are Stronger Knowledge Distillers](https://openaccess.thecvf.com/content/ICCV2023/html/Lao_Masked_Autoencoders_Are_Stronger_Knowledge_Distillers_ICCV_2023_paper.html) note: masked autoencoders = bert-like

- [3/10] (hyperbolic space operations end2end first network) [Poincare ResNet](https://openaccess.thecvf.com/content/ICCV2023/papers/van_Spengler_Poincare_ResNet_ICCV_2023_paper.pdf) maybe interesting in 5 years

### Encrypted inference

- (faster relu in ectrypted space) [AutoReP: Automatic ReLU Replacement for Fast Private Network Inference](https://openaccess.thecvf.com/content/ICCV2023/html/Peng_AutoReP_Automatic_ReLU_Replacement_for_Fast_Private_Network_Inference_ICCV_2023_paper.html) imagine you have hosting on cloud for the model but don't want google or whoever is maintaining this to know which data you sent/predictions obtained. so turns out there're networks which operate on encripted input and return encripted output. and ReLU in such representation is a bottleneck (~100x slower compared to convs lol). that's actually interesting alternative for infering user photos on backend with extra guarantees if needed (banks, governments, etc)

- (NAS for privacy-inference mode aware vits) [MPCViT: Searching for Accurate and Efficient MPC-Friendly Vision Transformer with Heterogeneous Attention](https://openaccess.thecvf.com/content/ICCV2023/html/Zeng_MPCViT_Searching_for_Accurate_and_Efficient_MPC-Friendly_Vision_Transformer_with_ICCV_2023_paper.html)

## Video

### Video Generation

- (no SD training. first frame latent -> denoise to an extent -> warp with camera motions through time -> noise again per frame -> denoise completely. +for complete denoising modified self-attn to work for every frame on first frame keys/values) [Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://openaccess.thecvf.com/content/ICCV2023/html/Khachatryan_Text2Video-Zero_Text-to-Image_Diffusion_Models_are_Zero-Shot_Video_Generators_ICCV_2023_paper.html) again old one

### Video + Audio

- [7/10] (audio->video, works on ~backgrounds) [The Power of Sound (TPoS): Audio Reactive Video Generation with Stable Diffusion](https://ku-vai.github.io/TPoS/) just curious: can it generate youtube video loop footage from background music? if quality is good enough maybe that's practical, if no:(

- [7/10] [Video Background Music Generation: Dataset, Method and Evaluation](https://openaccess.thecvf.com/content/ICCV2023/html/Zhuo_Video_Background_Music_Generation_Dataset_Method_and_Evaluation_ICCV_2023_paper.html)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/5f9cdbd4-66cc-4bd1-be3a-f5927f6e24dc)

### Video Editing

- [*] [8/10] [StableVideo: Text-driven Consistency-aware Diffusion Video Editing](https://openaccess.thecvf.com/content/ICCV2023/html/Chai_StableVideo_Text-driven_Consistency-aware_Diffusion_Video_Editing_ICCV_2023_paper.html) [code](https://github.com/rese1f/StableVideo) improvements over text2live for atlas-based stylization, probably has potential (need further reading)

- [3/10] (driving video + ref -> edit. based on gans so likely outdated) [VidStyleODE Disentangled Video Editing via StyleGAN and NeuralODEs](https://cyberiada.github.io/VidStyleODE/) likely not very practical + outdated. clip consistency loss between close frames is interesting regularization but likely introduces some error on it's own

### Video Stylization

- [3/10] (video stylization. train depth-conditioned video generator with extra temporal blocks -> infer on real video depth + edit prompt) [Runway GEN1](https://arxiv.org/pdf/2302.03011.pdf) just one more reminder how outdated is iccv research...

### Video Tagging

- [4/10] [Order-Prompted Tag Sequence Generation for Video Tagging](https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Order-Prompted_Tag_Sequence_Generation_for_Video_Tagging_ICCV_2023_paper.html)

## Other problems

### Vector graphics

- [7/10] (somehow converted images to sketch vectors with differential renderer) [CLIPascene: Scene Sketching with Different Types and Levels of Abstraction](https://openaccess.thecvf.com/content/ICCV2023/html/Vinker_CLIPascene_Scene_Sketching_with_Different_Types_and_Levels_of_Abstraction_ICCV_2023_paper.html)

- [6/10] (some differentiable vector graphics) [A Theory of Topological Derivatives for Inverse Rendering of Geometry](https://openaccess.thecvf.com/content/ICCV2023/html/Mehta_A_Theory_of_Topological_Derivatives_for_Inverse_Rendering_of_Geometry_ICCV_2023_paper.html)

### Style transfer

- [6/10] (fix vs repetative patterns (see img)) [AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks](https://openaccess.thecvf.com/content/ICCV2023/html/Hong_AesPA-Net_Aesthetic_Pattern-Aware_Style_Transfer_Networks_ICCV_2023_paper.html) quality is bad, but maybe this trick with fixing repetetive pattern is applicable elsewhere, e.g. for patch discriminators in GANs if they're made shallow

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/c44fd928-18b8-4651-ab89-999c13c2f4ba)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/d1c6b193-75cf-49f7-90db-81e00b92eb2f)

- [5/10] (style transfer: (content img, text description of emotional feeling)->stylized. new text-image dataset of emotional descriptions used as refs to train the model + some sophisticated losses) [Affective Image Filter: Reflecting Emotions from Text to Images](https://openaccess.thecvf.com//content/ICCV2023/papers/Weng_Affective_Image_Filter_Reflecting_Emotions_from_Text_to_Images_ICCV_2023_paper.pdf) interesting new problem, not sure about practical application

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/5f45820d-9dbe-482a-9018-6a50169ba718)

- [3/10] (style transfer from text ref. results not impressive at all) [StylerDALLE: Language-Guided Style Transfer Using
a Vector-Quantized Tokenizer of a Large-Scale Generative Model](https://openaccess.thecvf.com/content/ICCV2023/html/Xu_StylerDALLE_Language-Guided_Style_Transfer_Using_a_Vector-Quantized_Tokenizer_of_a_ICCV_2023_paper.html)

### 3D

- [*] [8/10] (nerf + sketch modification on 2+ views + modified prompt -> modified 3d model. sds with new prompt + regularization to preserve prev outout + loss to match masks for sketched img)[SKED: Sketch-guided Text-based 3D Editing](https://openaccess.thecvf.com/content/ICCV2023/html/Mikaeili_SKED_Sketch-guided_Text-based_3D_Editing_ICCV_2023_paper.html) likely should work for other editing tasks (not just 3d)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/3a537775-c9c8-430c-83e9-c28f5c9435fe)

- [*] [8/10] (base nerf + it's dataset -> edited nerf. render train view img -> edit with instruct pix2pix -> update train dataset image with it -> continue training nerf) [Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions](https://openaccess.thecvf.com/content/ICCV2023/html/Haque_Instruct-NeRF2NeRF_Editing_3D_Scenes_with_Instructions_ICCV_2023_paper.html) quality is very rough, e.g. identity is not preserved on man or night-mode on scene preserves clouds. smth else than 3d by same method? iterative dataset refinement paired with consistency looks like a decent idea

- [*] [8/10] (1)detr + sam to segment masks of areas of importance to be edited 2)stylization via base nerf loss + feature matching loss (vgg features of optimized nerf -> vgg features of nearest neighbour in vgg features of style img compared to unedited nerf pixel location) [S2RF: Semantically Stylized Radiance Fields](https://dishanil.github.io/S2RF/)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/d66c622c-80b7-425b-94a9-9f0a348ae391)

- [4/10] (distill nerfs so they're realtime on devices. quality likely drops) [Re-ReND: Real-Time Rendering of NeRFs across Devices](https://openaccess.thecvf.com/content/ICCV2023/html/Rojas_Re-ReND_Real-Time_Rendering_of_NeRFs_across_Devices_ICCV_2023_paper.html)

### image2image

- [*] [8/10] (problem: real backgrounds -> anime backgrounds with unpaired data. solution: 1)tune stylegan pre-trained on real background images on anime with clip & lpips consistency vs original source images 2)generate synthetic data AND filter them through segmentation consistency check 3)train on combined paired synthetic data and real unpaired photos/anime references. data: 90k real set, 6k real anime backgrounds from Makoto Sinkai's movies, 30k synth data (unkonown amount left after filtration). results look not great but ok ) [Scenimefy: Learning to Craft Anime Scene via Semi-Supervised Image-to-Image Translation](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Scenimefy_Learning_to_Craft_Anime_Scene_via_Semi-Supervised_Image-to-Image_Translation_ICCV_2023_paper.pdf) the approach is sophisticated but it does make sense in my eyes, esp with some modifications. likes: overall paired synth&unpaired real training idea, filtering by segmentation consistency, clip/lpips vs source while stylegan tuning (it works opposite way from stylization but does help consistency). not sure about patch losses (e.g. they can select similar patches with some chance, esp with large batch)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/84312129-0e4e-4f8a-a152-a4f5eb0c0509)

- [4/10] (image2image harmonization, experiments on realistic images only) [Deep Image Harmonization with Globally Guided Feature Transformation and
Relation Distillation](https://openaccess.thecvf.com/content/ICCV2023/html/Niu_Deep_Image_Harmonization_with_Globally_Guided_Feature_Transformation_and_Relation_ICCV_2023_paper.html) code/model won't be released but maybe useful if you want to train your image2image (although maybe just training on public/private datasets is sufficient)

### Inpainting

- [7/10] (LaMa clip has problems which prevents it from drawing non-regular texture. this paper fixes that module) [Rethinking Fast Fourier Convolution in Image Inpainting](https://openaccess.thecvf.com/content/ICCV2023/html/Chu_Rethinking_Fast_Fourier_Convolution_in_Image_Inpainting_ICCV_2023_paper.html) this op gives global context so might be useful in other architectures

- [5/10] (mobile inpainting - ~300ms IPhone 14 Pro. quality on their imgs looks ok. depthwise convs + resnet rgb prediction on multiple resolutions + overparametrization) [MI-GAN: A Simple Baseline for Image Inpainting on Mobile Devices](https://openaccess.thecvf.com/content/ICCV2023/html/Sargsyan_MI-GAN_A_Simple_Baseline_for_Image_Inpainting_on_Mobile_Devices_ICCV_2023_paper.html) I was surprised overparametrization helps that much

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/107cc237-e276-4380-aa42-cd2eca11fbea)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/f2266f2a-4646-4565-ab83-adcc3d1bf462)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/2930c24c-f547-4345-a1ce-758f91655097)

### Face Recognition

- [*] [6/10] (98% purely synthetic face recognition vs 99.8% real on some benchmark. (prod quality is smth like 99.99+ now in [top solutions](https://pages.nist.gov/frvt/html/frvt11.html)?). basically conditioned diffusion on face recognition embeddings) [IDiff-Face: Synthetic-based Face Recognition through Fizzy Identity-Conditioned Diffusion Model](https://openaccess.thecvf.com/content/ICCV2023/html/Boutros_IDiff-Face_Synthetic-based_Face_Recognition_through_Fizzy_Identity-Conditioned_Diffusion_Model_ICCV_2023_paper.html) there's chiken & egg problem: if you don't have baseline FR you can't condition diffusion, if you don't have diffusion you can't produce FR model. (so quality of your conditions depend on FR model baseline and by definition can't be higher). still good to know that such simple conditioning works (although they trained on really really close faces)

### Hair simulation/editing/animation

- [8/10] [HairCLIPv2: Unifying Hair Editing via Proxy Feature Blending](https://openaccess.thecvf.com/content/ICCV2023/html/Wei_HairCLIPv2_Unifying_Hair_Editing_via_Proxy_Feature_Blending_ICCV_2023_paper.html) code available. didn't check details but looks quite impressive

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/d860f2b5-b339-4c15-b16e-a2151f606e88)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/61ada8dc-93d8-44aa-aff9-daef9ec81764)

- [6/10] [Automatic Animation of Hair Blowing in Still Portrait Photos](https://openaccess.thecvf.com/content/ICCV2023/html/Xiao_Automatic_Animation_of_Hair_Blowing_in_Still_Portrait_Photos_ICCV_2023_paper.html) animations do not look natural for video but there're no individual image level artifacts, might be useful

- [5/10] (hair geometry reconstruction at a strand level from a monocular video or multi-view images captured in uncontrolled lighting conditions) [Neural Haircut: Prior-Guided Strand-Based Hair Reconstruction](https://openaccess.thecvf.com/content/ICCV2023/html/Sklyarova_Neural_Haircut_Prior-Guided_Strand-Based_Hair_Reconstruction_ICCV_2023_paper.html) quality is quite good

### Detection

- [ASAG: Building Strong One-Decoder-Layer Sparse Detectors via Adaptive Sparse Anchor Generation](https://openaccess.thecvf.com/content/ICCV2023/html/Fu_ASAG_Building_Strong_One-Decoder-Layer_Sparse_Detectors_via_Adaptive_Sparse_Anchor_ICCV_2023_paper.html)

### Adaptation to the unknown

- [9/10] (normally there's only known objects (labelled). at some point researchers took ROIs with high confidence and add label as unknown objects / use part of data this way - but it does not really generalize because of limited training data. in this work random boxes are sampled during training -> roi extracted -> matching loss on these boxes which "encourages exploration") [Random Boxes Are Open-world Object Detectors](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Random_Boxes_Are_Open-world_Object_Detectors_ICCV_2023_paper.html) I like this paper because it's a way for network to go beyound training data, at least find things it doesn't know but which LOOK interesting. w/o this we're just forcing network to memorize train (and there're always mistakes, most influential papers of conference all say just how important your data quality is)

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/1c3a228e-b5e5-4296-984b-77c235de0d0f)

- [5/10] (self-supervised object discovery. using pretrained dino to extract features -> foreground/background by simple criteria -> spatial clustering of foreground pixels -> form bboxes -> profit) [MOST: Multiple Object Localization with Self-Supervised Transformers for Object Discovery](https://openaccess.thecvf.com/content/ICCV2023/html/Rambhatla_MOST_Multiple_Object_Localization_with_Self-Supervised_Transformers_for_Object_Discovery_ICCV_2023_paper.html)

- [5/10] (some way to detect unknown objects & adapt to new distribution while training only on known objects) [Activate and Reject: Towards Safe Domain Generalization under Category Shift](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Activate_and_Reject_Towards_Safe_Domain_Generalization_under_Category_Shift_ICCV_2023_paper.html)

- [4/10] (segmentation: training on source domain + target domain descriptions -> inference on these unseen target domains. see diagram for details but basically feature augs via clip towards target domain. quality is better than 1-shot) [PODA: Prompt-driven Zero-shot Domain Adaptation](https://openaccess.thecvf.com/content/ICCV2023/html/Fahes_PODA_Prompt-driven_Zero-shot_Domain_Adaptation_ICCV_2023_paper.html) there're some extreme examples like driving through fire - if you have smth like this might be useful, otherwise labelling even few images is way better

![image](https://github.com/asmekal/iccv-2023-notes/assets/14358106/89137fae-5671-447c-9e2c-a10bc20bf7ef)

## Misc

- [7/10] (anime inbetweening [sketches]. new small dataset & [code](https://github.com/lisiyao21/AnimeInbet)) [Deep Geometrized Cartoon Line Inbetweening](https://openaccess.thecvf.com/content/ICCV2023/html/Siyao_Deep_Geometrized_Cartoon_Line_Inbetweening_ICCV_2023_paper.html) the dataset obtained in synthetic way (blender 3d models). that's important problem, which should be solved relatively easily, just need the data. my guess is anime studios are long working on it because it's the easiest and most straightforward thing in animation pipeline to be optimized. and it's sort of solved for video interpolation already so really - only need the data. this is the first work on the topic surprisingly

- [6/10] [Story Visualization by Online Text Augmentation with Context Memory](https://openaccess.thecvf.com/content/ICCV2023/html/Ahn_Story_Visualization_by_Online_Text_Augmentation_with_Context_Memory_ICCV_2023_paper.html) I just like the problem. Quality is looking far from good for now

- [6/10] (how humans selected names for colors? why the colors are chosen this way? this paper analyses natural world colors distribution and tries to go from few colors by splitting 1 color into 2 many times and comes to close to natural evolution. naming of colors is not considered as it is completely random but the colors themselves are relatively similar) [Name Your Colour For the Task: Artificially Discover Colour Naming via Colour Quantisation Transformer](https://openaccess.thecvf.com/content/ICCV2023/html/Su_Name_Your_Colour_For_the_Task_Artificially_Discover_Colour_Naming_ICCV_2023_paper.html) great idea, might be useful to predict future colors by extrapolation / colors of alien planets in games, etc

- [4/10] (artistic text by diffusion) [DS-Fusion: Artistic Typography via Discriminated and Stylized Diffusion](https://openaccess.thecvf.com/content/ICCV2023/html/Tanveer_DS-Fusion_Artistic_Typography_via_Discriminated_and_Stylized_Diffusion_ICCV_2023_paper.html) [this one](https://arxiv.org/pdf/1905.01354.pdf) is better imo

- [DLT: Conditioned layout generation with Joint Discrete-Continuous Diffusion Layout Transformer](https://openaccess.thecvf.com/content/ICCV2023/html/Levi_DLT_Conditioned_layout_generation_with_Joint_Discrete-Continuous_Diffusion_Layout_Transformer_ICCV_2023_paper.html) exists

- [Guided Motion Diffusion for Controllable Human Motion Synthesis](https://openaccess.thecvf.com/content/ICCV2023/html/Karunratanakul_Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis_ICCV_2023_paper.html) exists
