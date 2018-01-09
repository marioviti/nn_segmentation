from DataManager import DataManager

def main():
#    path = "../CD_Dataset"
#    dataloader = DataLoader(path)
#    train_w, train_x, train_y, eval_x, eval_y = dataloader.get_data()
#    dataholder = DataHolder(train_x, train_y,
#                            train_w= train_w,
#                            eval_x=eval_x, eval_y=eval_y)
#    datasampler = DataSampler(dataholder)
#    h,w = 350,350
#    batchs = datasampler.get_batch_patches(h,w,
#                                        as_images=True,
#                                        from_images=4, batch_size=5 )
#    datashower = DataShower()
#    datashower.show_batches(batchs,['X','Y','W'])
    datamanager = DataManager()
    datamanager.show_train_batch_patches(350,350,from_images=2, batch_size=5)


if __name__ == '__main__':
    main()
