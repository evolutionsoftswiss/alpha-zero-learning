package ch.evolutionsoft.rl.alphazero.connectfour.playground;

/**
 * 
 * @author evolutionsoft
 *
 */
public class Field{
    
    int position;
    
    public Field(int position) {
    	
        this.position = position;
    }

    public Field(String fieldString) {
    	
        if (!isValidFieldString(fieldString))
            throw new IllegalArgumentException("Invalid String for a field.");
        
        char columnChar = fieldString.charAt(0);
        char rowChar = fieldString.charAt(1);
        
        if (Character.isLowerCase(columnChar)) {
         
        	this.position = (rowChar - '0') * ArrayPlaygroundConstants.COLUMN_COUNT  + (columnChar - 'a') + 1;
        }
        else{
         
        	this.position = (rowChar - '0') * ArrayPlaygroundConstants.COLUMN_COUNT  + (columnChar - 'A') + 1;
        }
    }
    
    public int getPosition() {
    	
    	return this.position;
    }
    
    
    public int getColumn(){
    	
    	return this.position % ArrayPlaygroundConstants.COLUMN_COUNT - 1;
    }
    
    
    public int getRow(){
    	
    	return this.position / ArrayPlaygroundConstants.COLUMN_COUNT - 1;
    }
    
    
    public boolean equals(Object object) {
        if (object instanceof Field) 
            return this.equals((Field)object);
        if (object instanceof String)
            return this.equals((String)object);
        return false;
    }
    
    public boolean equals(Field otherField) {
    	
        return this.position == otherField.position;
    }
    
    public boolean equals(String fieldString) {
    	
        if (fieldString.length() == 2) {
        	
            char columnChar = fieldString.charAt(0);
            char rowChar = fieldString.charAt(1);
            
            if (Character.isLowerCase(columnChar)){
            	
            	return this.position / ArrayPlaygroundConstants.COLUMN_COUNT == (rowChar - '0') &&
            	       this.position % ArrayPlaygroundConstants.COLUMN_COUNT == (columnChar -'a') + 1;
            }
            
            if (Character.isUpperCase(columnChar)){
            	
            	return this.position / ArrayPlaygroundConstants.COLUMN_COUNT == (rowChar - '0') &&
            	       this.position % ArrayPlaygroundConstants.COLUMN_COUNT == (columnChar -'A') + 1;
            }
        }
        return false;
    }

    protected static boolean isValidFieldString(String fieldString) {
        if (fieldString.length() == 2) {
            char firstChar = fieldString.charAt(0);
            char secondChar = fieldString.charAt(1);
            if (Character.isLowerCase(firstChar))
                return firstChar >= 'a' && firstChar <= 'g'
                    && secondChar >= '1' && secondChar <= '6';
            return firstChar >= 'A' && firstChar <= 'G'
                && secondChar >= '1' && secondChar <= '6';
        }
        return false;
    }
}
