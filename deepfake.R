
# Preprocess Function

library(dplyr)
library(magrittr)
library(opencv)
library(magick)

data_dir = "~/Downloads/deepfake-detection-challenge/"

# Total files in train and test dirs
all_files = list.files(data_dir, pattern = ".mp4", recursive = TRUE) 

paste('Total video files',all_files %>% length())
paste('Total video files in train data', grep('train', all_files) %>% length())
paste('Total video files in test data', grep('test', all_files) %>% length())

# Preprocess and get frames
train_files = grep('train', all_files, value = TRUE)

# Read Metadata and find fakes and real ones

js = jsonlite::fromJSON('/home/turgut/Downloads/metadata.json')
train_ = data.frame('video_id'=names(js),
                    'video_id_2'=train_files,
                    'labels' = sapply(1:length(js), function(x) js[[x]][["label"]]))

train_$labels %>% table()
train_$labels %>% table() %>% prop.table()

# Take sample
train_ %>% filter(labels=='REAL') %>% 
  rbind(sample_n(train_ %>% filter(labels=='FAKE'),
                 nrow(train_ %>% filter(labels=='REAL') ))) -> train_

train_files = train_$video_id_2

t = Sys.time()
data_preprocess <- function(dir_name) {
  
  dir.create(dir_name)
  for (i in 1:length(train_files)) {
    video = image_read_video(paste(data_dir,train_files[i],sep=''), fps = 1)
    idx = ceiling(1/length(video)*100)
    
    list_of_frames = list()
    
    for (j in 1:idx) {
      vid_x = video[[j]] %>% image_read() %>% image_resize('1000x1000')
      list_of_frames[[j]]<-vid_x
      
      dir_n = gsub(train_files[i],replacement = '',pattern = 'train_sample_videos/|\\.mp4')
      
      if(!dir.exists(paste(dir_name,dir_n,sep = '/'))) {
        dir.create(paste(dir_name,dir_n,sep = '/'))
      }
      invisible(lapply(1:length(list_of_frames), 
                       function(x) image_write(list_of_frames[[x]],
                                               path = paste(paste(dir_name,'/', dir_n,'/', sep = ''), 'frame',j, '.png',sep = '_'))))
    }
    rm(video,list_of_frames)
    unlink(paste0(normalizePath(tempdir()), "/", dir(tempdir())), recursive = TRUE)
    gc()
    print(paste(dir_n,'which is',i, 'video out of',length(train_files)))
  }
  
}

data_preprocess('train_deep_fake')

Sys.time() - t

# Now, let's using opencv detect faces
# and then classify images into 2 groups:
# real and fake ones

train_ = train_ %>% mutate(dirr = gsub(video_id,replacement = '',pattern = '\\.mp4'))

nms_dirs = dir('train_deep_fake')
train_ = train_ %>% filter(dirr %in% nms_dirs)

fakes <- train_ %>% filter(labels=='FAKE')
reals <- train_ %>% filter(labels=='REAL')

fakes_dir <- paste('train_deep_fake/',fakes$dirr,sep='')
reals_dir <- paste('train_deep_fake/',reals$dirr,sep='')

data_prepare_faces <- function(dirs, new_dir_name) {
  
  dirs = dirs
  dir.create(new_dir_name)
  for(i in 1:length(dirs)){
    dir_idx = dirs[i]
    for (j in 1:length(list.files(dir_idx))) {
      filess = list.files(dir_idx)
      path_img = paste(dir_idx,'/',filess[j],sep = '')
      unconf <- ocv_read(path_img)
      faces <- ocv_face(unconf)
      facemask <- ocv_facemask(unconf)
      df = attr(facemask, 'faces')
      rectX = (df$x - df$radius) 
      rectY = (df$y - df$radius)
      x = (df$x + df$radius) 
      y = (df$y + df$radius)
      if(!nrow(df)==0){
        for (z in 1:nrow(df)) {
          imh  = image_read(path_img)
          edited = image_crop(imh, paste(x-rectX[z]+1,'x',x-rectX[z]+1,'+',rectX[z], '+',rectY[z],sep = ''))
          edited
          image_write(edited,paste(new_dir_name,'/','fake_',sample(1:10000,1),sample(1:10000,1),z,sep = ''))
        }
      }
    }
    print(paste('Done',i,'out of',length(dirs)))
  }
}

data_prepare_faces(fakes_dir,'fakes')
data_prepare_faces(reals_dir,'reals')

library(ff)
dir.create('fakes_reals')
dir.create('fakes_reals/reals')
dir.create('fakes_reals/fakes')

from_r <- "~/Downloads/reals"           
to_r   <- "~/Downloads/fakes_reals/reals"
from_f <- "~/Downloads/fakes"           
to_f   <- "~/Downloads/fakes_reals/fakes"
file.move(from_r,to_r)
file.move(from_f,to_f)

files_to_png = list.files('~/Downloads/fakes_reals',recursive = T,full.names = T)
sapply(1:length(files_to_png), function(x) file.rename(files_to_png[x], paste(files_to_png[x],'.png',sep = '')))

library(keras)
library(tensorflow)

tensorflow::tf_version()
physical_devices = tf$config$list_physical_devices('GPU')
tf$config$experimental$set_memory_growth(physical_devices[[1]],T)

train_dir = '~/Downloads/fakes_reals'
width = 150L
height = 150L
epochs = 10

train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest",
  validation_split=0.2
)


train_generator <- flow_images_from_directory(
  train_dir,                  
  train_datagen,             
  target_size = c(width,height), 
  batch_size = 10,
  class_mode = "binary"
)

# Build the model ---------------------------------------------------------

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(width, height, 3)
)

model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = ceiling(train_generator$samples/train_generator$batch_size),
  epochs = 10
)