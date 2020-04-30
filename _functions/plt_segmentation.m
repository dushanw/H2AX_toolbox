function count = plt_segmentation(I,BW,BW_gt,name_stem)
      
      [L,count.pred] =bwlabel(BW);
      STATS     = regionprops(L, 'Centroid');
      B         = bwboundaries(BW);
      
      h = imshow(I);
      hold on
      for(k=1:count.pred)
          c = round(STATS(k,1).Centroid(1));
          r = round(STATS(k,1).Centroid(2));
              
          plot(c,r,'+g','MarkerSize',6,'LineWidth',1);                                  
      end
      
      if (size(B,1)>0)
          for(i=1:size(B,1))
                Outline = B{i};
                plot(Outline(:,2),Outline(:,1),'b','LineWidth',1);
          end
      end
      if ~isempty(BW_gt)         
          [L,count.gt] =bwlabel(BW_gt);
          STATS     = regionprops(L, 'Centroid');
          B         = bwboundaries(BW_gt);                    
          if (size(B,1)>0)
              for(i=1:size(B,1))
                    Outline = B{i};
                    plot(Outline(:,2),Outline(:,1),'r','LineWidth',1);
              end
          end          
      end
      hold off
      
      saveas(h,[name_stem '.jpeg']);      
end