
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

fakes <- train_ %>% filter(labels=='FAKE')
reals <- train_ %>% filter(labels=='REAL')

#list_of_dirs <- list.dirs('~/Downloads/train_deep_fake',full.names = FALSE)


fakes_dir <- paste('train_deep_fake/',fakes$dirr,sep='')
reals_dir <- paste('train_deep_real/',reals$dirr,sep='')

data_prepare_faces <- function(dirs,new_dir_name) {
  
  dirs = dirs
  dir.create(new_dir_name)
  for(i in 1:dirs){
    dir_idx = dirs[i]
    for (j in 1:length(list.files(dir_idx))) {
      filess = list.files(dir_idx)
      unconf <- ocv_read(paste(dir_idx,'/',filess[j],sep = ''))
      faces <- ocv_face(unconf)
      facemask <- ocv_facemask(unconf)
      df = attr(facemask, 'faces')
      rectX = (df$x - df$radius) 
      rectY = (df$y - df$radius)
      x = (df$x + df$radius) 
      y = (df$y + df$radius)
      
      edited = image_crop(imh, "49x49+66+34")
      edited = image_crop(imh, paste(x-rectX+1,'x',x-rectX+1,'+',rectX, '+',rectY,sep = ''))
      edited
      image_write(edited,'/home/turgut/Downloads/video/frame_1_face.png')
    }
  }
}



unconf <- ocv_read('/home/turgut/Downloads/video/frame_1.png')
faces <- ocv_face(unconf)
faces
#ocv_write(faces, 'faces.jpg')
facemask <- ocv_facemask(unconf)
df = attr(facemask, 'faces')
rectX = (df$x - df$radius) 
rectY = (df$y - df$radius)
x = (df$x + df$radius) 
y = (df$y + df$radius)
#crop_img = self.img[y:(y+2*r), x:(x+2*r)]

imh  = image_draw(image_read('/home/turgut/Downloads/video/frame_1.png'))
rect(rectX, rectY, x, y, border = "red", lty = "dashed", lwd = 2)
dev.off()

image_info(imh)->info
x-rectX+1
y-rectY+1
df

edited = image_crop(imh, "49x49+66+34")
edited = image_crop(imh, paste(x-rectX+1,'x',x-rectX+1,'+',rectX, '+',rectY,sep = ''))
edited
image_write(edited,'/home/turgut/Downloads/video/frame_1_face.png')

